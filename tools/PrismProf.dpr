program PrismProf;

{ Misst, wohin die Zeit je Token geht.

  Prism.Vector haelt den GPU-Pfad hinter dem globalen Funktionszeiger
  QMatVecHook. Den haengen wir um: erst mitstoppen, dann an den echten
  Vulkan-Pfad weiterreichen. Damit faellt die Aufteilung ab, ohne dass an
  Prism selbst etwas geaendert werden muss.

    Wanduhr = Zeit IN MatVec + Zeit AUSSERHALB (Norm, RoPE, Attention, SiLU)

  Danach dieselben Tensoren noch einmal Rueck an Rueck ohne CPU-Arbeit
  dazwischen -- das trennt die reinen Versandkosten von dem, was die
  Verzahnung mit der CPU zusaetzlich kostet. }

{$APPTYPE CONSOLE}
{$POINTERMATH ON}

uses
  System.SysUtils,
  System.Classes,
  System.Diagnostics,
  System.Generics.Collections,
  Prism.Types in '..\src\Prism.Types.pas',
  Prism.Vector in '..\src\Prism.Vector.pas',
  Prism.Tokenizer in '..\src\Prism.Tokenizer.pas',
  Prism.Model in '..\src\Prism.Model.pas',
  Prism.Streaming in '..\src\Prism.Streaming.pas',
  Prism.Vulkan.Api in '..\src\Prism.Vulkan.Api.pas',
  Prism.Vulkan in '..\src\Prism.Vulkan.pas',
  Prism.Gpu in '..\src\Prism.Gpu.pas',
  Prism.Gguf in '..\src\Prism.Gguf.pas',
  Prism.Llama in '..\src\Prism.Llama.pas',
  Prism.Inference in '..\src\Prism.Inference.pas';

type
  TShapeStat = record
    Rows, Cols: Integer;
    Calls: Int64;
    Ticks: Int64;
  end;

var
  GInner: TQMatVecHook = nil;
  GTicks: Int64 = 0;
  GCalls: Int64 = 0;
  GStats: TList<TShapeStat>;
  GOn: Boolean = False;

function ProfHook(const T: TQTensor; Y, X: PSingle): Boolean;
var
  T0: Int64;
  D: Int64;
  I: Integer;
  S: TShapeStat;
begin
  if not GOn then
    Exit(GInner(T, Y, X));
  T0 := TStopwatch.GetTimeStamp;
  Result := GInner(T, Y, X);
  D := TStopwatch.GetTimeStamp - T0;
  Inc(GTicks, D);
  Inc(GCalls);
  for I := 0 to GStats.Count - 1 do
    if (GStats[I].Rows = T.Rows) and (GStats[I].Cols = T.Cols) then
    begin
      S := GStats[I];
      Inc(S.Calls);
      Inc(S.Ticks, D);
      GStats[I] := S;
      Exit;
    end;
  S.Rows := T.Rows;
  S.Cols := T.Cols;
  S.Calls := 1;
  S.Ticks := D;
  GStats.Add(S);
end;

function Ms(Ticks: Int64): Double;
begin
  Result := Ticks * 1000.0 / TStopwatch.Frequency;
end;

var
  ModelPath: string;
  Ctx, NTok: Integer;
  GpuInfo: string;
  Backend: TLlamaBackend;
  Gen: TGenerator;
  SP: TSamplingParams;
  Usage: TUsage;
  Prompt: TArray<Integer>;
  Msgs: TChatMessages;
  Wall: TStopwatch;
  WallMs, InMs, OutMs: Double;
  Txt: string;
  I, L: Integer;
  S: TShapeStat;
  Lay: TLlamaLayer;
  Xb, Yb: TArray<Single>;
  Reps, R: Integer;
  T0: Int64;
  B2B: Double;
  MaxDim: Integer;

begin
  try
    ModelPath := ParamStr(1);
    Ctx := StrToIntDef(ParamStr(2), 2048);
    NTok := StrToIntDef(ParamStr(3), 40);
    GStats := TList<TShapeStat>.Create;

    if not TryInitGpuBackend(GpuInfo, 0,
         procedure(S: string) begin Writeln('  ', S); end) then
    begin
      Writeln('Keine GPU: ', GpuInfo);
      Halt(1);
    end;
    Writeln('GPU: ', GpuInfo);

    { Umhaengen NACH der GPU-Anschaltung -- die installiert den echten Haken. }
    GInner := QMatVecHook;
    if not Assigned(GInner) then
    begin
      Writeln('QMatVecHook ist nicht gesetzt -- laeuft der GPU-Pfad ueberhaupt?');
      Halt(1);
    end;
    QMatVecHook := ProfHook;

    Backend := TLlamaBackend.Create(ModelPath, Ctx, 0, nil);
    try
      Writeln('Modell: ', Backend.ModelName);
      Gen := TGenerator.Create(Backend);
      try
        SetLength(Msgs, 2);
        Msgs[0] := TChatMessage.Make('system', 'Du bist ein hilfreicher Assistent.');
        Msgs[1] := TChatMessage.Make('user',
          'Erklaere in etwa zehn Saetzen, wozu eine Rechnungsnummer dient.');
        Prompt := Backend.Tokenizer.BuildChatTokens(Msgs, Backend.DefaultTemplate);

        SP := TSamplingParams.Default;
        SP.MaxTokens := NTok;
        SP.Temperature := 0.7;

        { Aufwaermen: die Gewichte wandern beim ERSTEN Zugriff in den
          Grafikspeicher. Wer das mitmisst, misst das Hochladen. }
        Writeln('Aufwaermen...');
        SP.MaxTokens := 2;
        Gen.Generate(Prompt, SP, nil, Usage);

        SP.MaxTokens := NTok;
        GTicks := 0; GCalls := 0; GStats.Clear;
        GOn := True;
        Wall := TStopwatch.StartNew;
        Txt := Gen.Generate(Prompt, SP, nil, Usage);
        Wall.Stop;
        GOn := False;
        Writeln;
        Writeln('--- Antwort (zum Draufschauen, ob das Modell noch Sinn redet) ---');
        Writeln(Copy(Txt, 1, 400));

        WallMs := Wall.Elapsed.TotalMilliseconds;
        InMs := Ms(GTicks);
        OutMs := WallMs - InMs;

        Writeln;
        Writeln('=== Aufteilung ueber ', Usage.PromptTokens, ' Prompt- + ',
          Usage.CompletionTokens, ' Ausgabe-Token ===');
        Writeln(Format('Wanduhr gesamt        %10.1f ms', [WallMs]));
        Writeln(Format('  davon IN MatVec     %10.1f ms  (%.1f %%, %d Aufrufe)',
          [InMs, 100 * InMs / WallMs, GCalls]));
        Writeln(Format('  davon AUSSERHALB    %10.1f ms  (%.1f %%)  <- Norm, RoPE, Attention, SiLU, Sampling',
          [OutMs, 100 * OutMs / WallMs]));
        Writeln(Format('MatVec-Aufrufe je Token %8.1f', [GCalls /
          (Usage.PromptTokens + Usage.CompletionTokens)]));
        Writeln(Format('Kosten je MatVec        %8.3f ms', [InMs / GCalls]));

        Writeln;
        Writeln('=== nach Form (in der echten Erzeugung) ===');
        Writeln('     Rows x Cols   Aufrufe   ges. ms   je Auf. ms   Anteil');
        for I := 0 to GStats.Count - 1 do
        begin
          S := GStats[I];
          Writeln(Format('%9d x %-6d %8d %9.1f %11.3f %8.1f %%',
            [S.Rows, S.Cols, S.Calls, Ms(S.Ticks), Ms(S.Ticks) / S.Calls,
             100 * Ms(S.Ticks) / InMs]));
        end;

        { Rueck an Rueck: dieselben Tensoren, keine CPU-Arbeit dazwischen.
          Zeigt, was der reine Versand kostet, wenn die Karte nicht auf die
          CPU wartet. }
        MaxDim := 0;
        for I := 0 to GStats.Count - 1 do
        begin
          if GStats[I].Rows > MaxDim then MaxDim := GStats[I].Rows;
          if GStats[I].Cols > MaxDim then MaxDim := GStats[I].Cols;
        end;
        SetLength(Xb, MaxDim);
        SetLength(Yb, MaxDim);
        for I := 0 to MaxDim - 1 do
          Xb[I] := 0.01;

        Reps := 20;
        GOn := False;
        T0 := TStopwatch.GetTimeStamp;
        for R := 1 to Reps do
          for L := 0 to Backend.Model.Cfg.NLayers - 1 do
          begin
            Lay := Backend.Model.GetLayer(L);
            Lay.Wq.MatVec(@Yb[0], @Xb[0]);
            Lay.Wk.MatVec(@Yb[0], @Xb[0]);
            Lay.Wv.MatVec(@Yb[0], @Xb[0]);
            Lay.Wo.MatVec(@Yb[0], @Xb[0]);
            Lay.WGate.MatVec(@Yb[0], @Xb[0]);
            Lay.WUp.MatVec(@Yb[0], @Xb[0]);
            Lay.WDown.MatVec(@Yb[0], @Xb[0]);
          end;
        B2B := Ms(TStopwatch.GetTimeStamp - T0) / Reps;

        Writeln;
        Writeln('=== dieselben MatVecs Rueck an Rueck, ohne CPU dazwischen ===');
        Writeln(Format('ein Token-Aequivalent (%d Schichten x 7)  %8.1f ms',
          [Backend.Model.Cfg.NLayers, B2B]));
        Writeln(Format('in der echten Erzeugung dafuer            %8.1f ms',
          [InMs / (Usage.PromptTokens + Usage.CompletionTokens)]));
        Writeln;
        Writeln('GPU: ', GpuDetails);
      finally
        Gen.Free;
      end;
    finally
      Backend.Free;
    end;
  except
    on E: Exception do
    begin
      Writeln('EXCEPTION ', E.ClassName, ': ', E.Message);
      Halt(1);
    end;
  end;
end.
