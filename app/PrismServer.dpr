program PrismServer;

{ Prism REST-Server (Konsole, Windows/Linux/macOS).

  Eigenes Modell:
    PrismServer --model model\model.prism --tokenizer model\tokenizer.json --port 11434

  Existierendes GGUF-Modell (llama.cpp-Format):
    PrismServer --model models\tinyllama-1.1b-chat.Q8_0.gguf --ctx 1024

  Wichtige Optionen:
    --stream-layers N   Cluster-Streaming: nur N Layer im RAM (0 = alle)
    --experts-cache N   max. gecachte MoE-Experten (nur .prism)
    --ctx N             Kontextfenster kappen (GGUF, spart KV-Cache-RAM)
    --verify            Selbst-Verifikation standardmaessig aktiv
    --train             Online-Finetuning via POST /api/train (nur .prism)
    --template T        auto | prism | chatml | llama2 | plain
    --gpu               OpenCL-GPU-Backend versuchen }

{$APPTYPE CONSOLE}

uses
  System.SysUtils,
  System.Classes,
  Prism.Types in '..\src\Prism.Types.pas',
  Prism.Vector in '..\src\Prism.Vector.pas',
  Prism.Tensor in '..\src\Prism.Tensor.pas',
  Prism.Tokenizer in '..\src\Prism.Tokenizer.pas',
  Prism.Model in '..\src\Prism.Model.pas',
  Prism.Streaming in '..\src\Prism.Streaming.pas',
  Prism.Gpu in '..\src\Prism.Gpu.pas',
  Prism.Inference in '..\src\Prism.Inference.pas',
  Prism.Gguf in '..\src\Prism.Gguf.pas',
  Prism.Llama in '..\src\Prism.Llama.pas',
  Prism.Train in '..\src\Prism.Train.pas',
  Prism.Multimodal in '..\src\Prism.Multimodal.pas',
  Prism.Verify in '..\src\Prism.Verify.pas',
  Prism.RestServer in '..\src\Prism.RestServer.pas';

function ArgValue(const Name, Def: string): string;
var
  I: Integer;
begin
  Result := Def;
  for I := 1 to ParamCount - 1 do
    if SameText(ParamStr(I), '--' + Name) then
      Exit(ParamStr(I + 1));
end;

function HasArg(const Name: string): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if SameText(ParamStr(I), '--' + Name) then
      Exit(True);
end;

var
  Opts: TServerOptions;
  Server: TPrismRestServer;
  Line: string;

begin
  try
    Opts := TServerOptions.Default;
    Opts.ModelPath := ArgValue('model', '');
    Opts.TokenizerPath := ArgValue('tokenizer', '');
    Opts.Port := StrToIntDef(ArgValue('port', '11434'), 11434);
    Opts.StreamLayers := StrToIntDef(ArgValue('stream-layers', '0'), 0);
    Opts.MaxCachedExperts := StrToIntDef(ArgValue('experts-cache', '8'), 8);
    Opts.CtxOverride := StrToIntDef(ArgValue('ctx', '0'), 0);
    Opts.VerifyDefault := HasArg('verify');
    Opts.TrainEnabled := HasArg('train');
    Opts.CorpusPath := ArgValue('corpus', 'corpus.bin');
    Opts.Template := ArgValue('template', 'auto');
    Opts.UseGpu := HasArg('gpu');

    if Opts.ModelPath = '' then
    begin
      Writeln('Prism ', PRISM_VERSION, ' - LLM-Server (OpenAI/Ollama-kompatibel)');
      Writeln('Aufruf: PrismServer --model <datei.prism|datei.gguf> [Optionen]');
      Writeln('        siehe README.md fuer alle Optionen');
      Halt(1);
    end;

    Server := TPrismRestServer.Create(Opts,
      procedure(S: string)
      begin
        Writeln(FormatDateTime('hh:nn:ss', Now), '  ', S);
      end);
    try
      Server.Start;
      Writeln('Beenden mit "quit" + Enter (oder Strg+C).');
      while True do
      begin
        if Eof(Input) then
          { stdin geschlossen (Hintergrund-/Dienstbetrieb): weiterlaufen }
          while True do
            TThread.Sleep(1000);
        Readln(Line);
        if SameText(Trim(Line), 'quit') then
          Break;
      end;
    finally
      Server.Free;
    end;
  except
    on E: Exception do
    begin
      Writeln('FEHLER: ', E.Message);
      Halt(1);
    end;
  end;
end.
