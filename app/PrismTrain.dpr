program PrismTrain;

{ Prism Trainings-CLI.

  Workflow:
    1. PrismTrain tokenizer --corpus data\corpus.txt --vocab 512 --out model\tokenizer.json
    2. PrismTrain tokenize  --corpus data\corpus.txt --tokenizer model\tokenizer.json --out model\corpus.tokens
    3. PrismTrain init      --tokenizer model\tokenizer.json --dim 192 --layers 6 --heads 6 --seq 256 --experts 1 --out model\model.prism
    4. PrismTrain train     --model model\model.prism --tokens model\corpus.tokens --steps 2000 --batch 4 --seq 128 --lr 0.0003
    5. PrismTrain sample    --model model\model.prism --tokenizer model\tokenizer.json --chat --prompt "Was ist Delphi?" }

{$APPTYPE CONSOLE}

uses
  System.SysUtils,
  System.Classes,
  System.IOUtils,
  System.Diagnostics,
  Prism.Types in '..\src\Prism.Types.pas',
  Prism.Vector in '..\src\Prism.Vector.pas',
  Prism.Tensor in '..\src\Prism.Tensor.pas',
  Prism.Tokenizer in '..\src\Prism.Tokenizer.pas',
  Prism.Model in '..\src\Prism.Model.pas',
  Prism.Streaming in '..\src\Prism.Streaming.pas',
  Prism.Gpu in '..\src\Prism.Gpu.pas',
  Prism.Inference in '..\src\Prism.Inference.pas',
  Prism.Train in '..\src\Prism.Train.pas',
  Prism.Multimodal in '..\src\Prism.Multimodal.pas',
  Prism.Verify in '..\src\Prism.Verify.pas';

function ArgValue(const Name, Def: string): string;
var
  I: Integer;
begin
  Result := Def;
  for I := 1 to ParamCount - 1 do
    if SameText(ParamStr(I), '--' + Name) then
      Exit(ParamStr(I + 1));
end;

function ArgInt(const Name: string; Def: Integer): Integer;
begin
  Result := StrToIntDef(ArgValue(Name, ''), Def);
end;

function ArgFloat(const Name: string; Def: Double): Double;
var
  S: string;
  FS: TFormatSettings;
begin
  S := ArgValue(Name, '');
  if S = '' then
    Exit(Def);
  FS := TFormatSettings.Invariant;
  Result := StrToFloatDef(S, Def, FS);
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

procedure Say(const S: string);
begin
  Writeln(S);
end;

procedure CmdTokenizer;
var
  Tok: TTokenizer;
  Data: TBytes;
  OutPath: string;
begin
  Data := TFile.ReadAllBytes(ArgValue('corpus', 'corpus.txt'));
  OutPath := ArgValue('out', 'tokenizer.json');
  Tok := TTokenizer.Create;
  try
    Say(Format('Trainiere Byte-BPE (Zielvokabular %d) ...',
      [ArgInt('vocab', 512)]));
    Tok.TrainFromData(Data, ArgInt('vocab', 512),
      ArgInt('max-bytes', 8 * 1024 * 1024), procedure(S: string)
      begin
        Say('  ' + S);
      end);
    Tok.SaveToFile(OutPath);
    Say(Format('Tokenizer gespeichert: %s (Vokabular %d)',
      [OutPath, Tok.VocabSize]));
  finally
    Tok.Free;
  end;
end;

procedure CmdTokenize;
var
  Tok: TTokenizer;
  Data: TBytes;
  Tokens: TArray<Integer>;
  OutPath: string;
begin
  Tok := TTokenizer.Create;
  try
    Tok.LoadFromFile(ArgValue('tokenizer', 'tokenizer.json'));
    Data := TFile.ReadAllBytes(ArgValue('corpus', 'corpus.txt'));
    Say(Format('Tokenisiere %d Bytes ...', [Length(Data)]));
    Tokens := Tok.EncodeData(Data);
    OutPath := ArgValue('out', 'corpus.tokens');
    SaveTokensFile(OutPath, Tokens);
    Say(Format('%d Tokens -> %s (Kompression %.2fx)',
      [Length(Tokens), OutPath, Length(Data) / Length(Tokens)]));
  finally
    Tok.Free;
  end;
end;

procedure CmdInit;
var
  Tok: TTokenizer;
  W: TFullWeights;
  Cfg: TModelConfig;
  OutPath: string;
begin
  Tok := TTokenizer.Create;
  try
    Tok.LoadFromFile(ArgValue('tokenizer', 'tokenizer.json'));
    Cfg.VocabSize := Tok.VocabSize;
  finally
    Tok.Free;
  end;
  Cfg.SeqLen := ArgInt('seq', 256);
  Cfg.Dim := ArgInt('dim', 192);
  Cfg.NumLayers := ArgInt('layers', 6);
  Cfg.NumHeads := ArgInt('heads', 6);
  Cfg.NumExperts := ArgInt('experts', 1);
  if Cfg.Dim mod Cfg.NumHeads <> 0 then
  begin
    Say('FEHLER: dim muss durch heads teilbar sein.');
    Halt(1);
  end;
  W := TFullWeights.Create;
  try
    W.InitRandom(Cfg, UInt64(ArgInt('seed', 1337)));
    OutPath := ArgValue('out', 'model.prism');
    W.SaveToFile(OutPath);
    Say('Modell initialisiert: ' + OutPath);
    Say('  ' + Cfg.ToString);
    Say(Format('  Parameter: %.2f M (%.1f MB als F32)',
      [W.Layout.TotalCount / 1e6, W.Layout.TotalCount * 4 / 1024 / 1024]));
    if Cfg.IsMoE then
      Say(Format('  MoE aktiv: %d Experten/Layer, pro Token 1 aktiv',
        [Cfg.NumExperts]));
  finally
    W.Free;
  end;
end;

procedure CmdTrain;
var
  W: TFullWeights;
  Trainer: TTrainer;
  Tokens: TArray<Integer>;
  Rng: TRng;
  Steps, SaveEvery, S: Integer;
  LR, WD: Single;
  Loss, AvgLoss: Double;
  ModelPath: string;
  Watch: TStopwatch;
begin
  ModelPath := ArgValue('model', 'model.prism');
  W := TFullWeights.Create;
  try
    W.LoadFromFile(ModelPath);
    Say('Modell geladen: ' + W.Config.ToString);
    Tokens := LoadTokensFile(ArgValue('tokens', 'corpus.tokens'));
    Say(Format('Trainingsdaten: %d Tokens', [Length(Tokens)]));
    Trainer := TTrainer.Create(W, ArgInt('batch', 4), ArgInt('seq', 128));
    try
      Steps := ArgInt('steps', 500);
      SaveEvery := ArgInt('save-every', 100);
      LR := ArgFloat('lr', 3e-4);
      WD := ArgFloat('wd', 0.0);
      Rng.Seed(UInt64(ArgInt('seed', 42)));
      AvgLoss := 0;
      Watch := TStopwatch.StartNew;
      for S := 1 to Steps do
      begin
        Loss := Trainer.TrainStep(Tokens, Rng, LR, WD);
        if AvgLoss = 0 then
          AvgLoss := Loss
        else
          AvgLoss := 0.95 * AvgLoss + 0.05 * Loss;
        if (S mod 10 = 0) or (S = 1) then
          Say(Format('Schritt %5d/%d  Loss %.4f  (Mittel %.4f)  %.1f ms/Schritt',
            [S, Steps, Loss, AvgLoss,
             Watch.ElapsedMilliseconds / S]));
        if (SaveEvery > 0) and (S mod SaveEvery = 0) then
        begin
          W.SaveToFile(ModelPath);
          Say('  Checkpoint gespeichert.');
        end;
      end;
      W.SaveToFile(ModelPath);
      Say('Training abgeschlossen, Modell gespeichert: ' + ModelPath);
    finally
      Trainer.Free;
    end;
  finally
    W.Free;
  end;
end;

procedure CmdSample;
var
  W: TFullWeights;
  Tok: TTokenizer;
  BE: TPrismBackend;
  Gen: TGenerator;
  SP: TSamplingParams;
  Usage: TUsage;
  Tokens: TArray<Integer>;
  Msgs: TChatMessages;
begin
  W := TFullWeights.Create;
  W.LoadFromFile(ArgValue('model', 'model.prism'));
  Tok := TTokenizer.Create;
  Tok.LoadFromFile(ArgValue('tokenizer', 'tokenizer.json'));
  BE := TPrismBackend.Create(W, Tok, 'prism');
  try
    SP := TSamplingParams.Default;
    SP.Temperature := ArgFloat('temp', 0.8);
    SP.MaxTokens := ArgInt('max', 128);
    SP.TopK := ArgInt('top-k', 40);
    if HasArg('chat') then
    begin
      SetLength(Msgs, 1);
      Msgs[0] := TChatMessage.Make('user', ArgValue('prompt', 'Hallo!'));
      Tokens := Tok.BuildChatTokens(Msgs, ctPrism);
    end
    else
      Tokens := Tok.Encode(ArgValue('prompt', ''));
    Gen := TGenerator.Create(BE);
    try
      Gen.Generate(Tokens, SP,
        procedure(Chunk: string)
        begin
          Write(Chunk);
        end, Usage);
      Writeln;
      Say(Format('[%d Prompt- + %d Antwort-Tokens]',
        [Usage.PromptTokens, Usage.CompletionTokens]));
    finally
      Gen.Free;
    end;
  finally
    BE.Free; // gibt W und Tok frei
  end;
end;

var
  Cmd: string;

begin
  try
    Cmd := LowerCase(ParamStr(1));
    if Cmd = 'tokenizer' then
      CmdTokenizer
    else if Cmd = 'tokenize' then
      CmdTokenize
    else if Cmd = 'init' then
      CmdInit
    else if Cmd = 'train' then
      CmdTrain
    else if Cmd = 'sample' then
      CmdSample
    else
    begin
      Say('Prism ' + PRISM_VERSION + ' - Trainings-CLI');
      Say('Befehle:');
      Say('  tokenizer --corpus F --vocab N --out tokenizer.json');
      Say('  tokenize  --corpus F --tokenizer T --out corpus.tokens');
      Say('  init      --tokenizer T --dim N --layers N --heads N --seq N --experts N --out model.prism');
      Say('  train     --model M --tokens F --steps N --batch N --seq N --lr X');
      Say('  sample    --model M --tokenizer T [--chat] --prompt "..."');
    end;
  except
    on E: Exception do
    begin
      Writeln('FEHLER: ', E.Message);
      Halt(1);
    end;
  end;
end.
