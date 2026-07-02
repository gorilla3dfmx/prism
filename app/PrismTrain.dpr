program PrismTrain;

{ Prism training CLI.

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
  System.Math,
  System.Diagnostics,
  Prism.Types in '..\src\Prism.Types.pas',
  Prism.Vector in '..\src\Prism.Vector.pas',
  Prism.Tensor in '..\src\Prism.Tensor.pas',
  Prism.Tokenizer in '..\src\Prism.Tokenizer.pas',
  Prism.Model in '..\src\Prism.Model.pas',
  Prism.Streaming in '..\src\Prism.Streaming.pas',
  Prism.Gpu in '..\src\Prism.Gpu.pas',
  Prism.Laws in '..\src\Prism.Laws.pas',
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
    Say(Format('Training byte BPE (target vocabulary %d) ...',
      [ArgInt('vocab', 512)]));
    Tok.TrainFromData(Data, ArgInt('vocab', 512),
      ArgInt('max-bytes', 8 * 1024 * 1024), procedure(S: string)
      begin
        Say('  ' + S);
      end);
    Tok.SaveToFile(OutPath);
    Say(Format('Tokenizer saved: %s (vocabulary %d)',
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
    Say(Format('Tokenizing %d bytes ...', [Length(Data)]));
    Tokens := Tok.EncodeData(Data);
    OutPath := ArgValue('out', 'corpus.tokens');
    SaveTokensFile(OutPath, Tokens);
    Say(Format('%d tokens -> %s (compression %.2fx)',
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
    Say('ERROR: dim must be divisible by heads.');
    Halt(1);
  end;
  W := TFullWeights.Create;
  try
    W.InitRandom(Cfg, UInt64(ArgInt('seed', 1337)));
    OutPath := ArgValue('out', 'model.prism');
    W.SaveToFile(OutPath);
    Say('Model initialized: ' + OutPath);
    Say('  ' + Cfg.ToString);
    Say(Format('  Parameters: %.2f M (%.1f MB as F32)',
      [W.Layout.TotalCount / 1e6, W.Layout.TotalCount * 4 / 1024 / 1024]));
    if Cfg.IsMoE then
      Say(Format('  MoE active: %d experts/layer, 1 active per token',
        [Cfg.NumExperts]));
  finally
    W.Free;
  end;
end;

procedure CmdTrain;
var
  W: TFullWeights;
  Trainer: TTrainer;
  Domains: TArray<TArray<Integer>>;
  TokenFiles: TArray<string>;
  TotalLen, Pick: Int64;
  Rng: TRng;
  Steps, SaveEvery, S, D, I: Integer;
  LR, WD, RouterAux: Single;
  Loss, AvgLoss: Double;
  ModelPath: string;
  Watch: TStopwatch;
begin
  ModelPath := ArgValue('model', 'model.prism');
  W := TFullWeights.Create;
  try
    W.LoadFromFile(ModelPath);
    Say('Model loaded: ' + W.Config.ToString);
    { --tokens accepts a comma-separated list; with more than one file,
      file index = domain id = expert ("thematic area") and the router
      is guided by an auxiliary loss (--router-aux, default 0.1) }
    TokenFiles := ArgValue('tokens', 'corpus.tokens').Split([',']);
    SetLength(Domains, Length(TokenFiles));
    TotalLen := 0;
    for I := 0 to High(TokenFiles) do
    begin
      Domains[I] := LoadTokensFile(Trim(TokenFiles[I]));
      Inc(TotalLen, Length(Domains[I]));
      Say(Format('Training data [domain %d]: %s (%d tokens)',
        [I, Trim(TokenFiles[I]), Length(Domains[I])]));
    end;
    if Length(Domains) > 1 then
    begin
      RouterAux := ArgFloat('router-aux', 0.1);
      if not W.Config.IsMoE then
        Say('WARNING: multiple domain files but the model has no experts ' +
          '(init with --experts N).');
      if Length(Domains) > W.Config.NumExperts then
        Say('WARNING: more domain files than experts; domain id will wrap.');
    end
    else
      RouterAux := ArgFloat('router-aux', 0.0);
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
        { pick a domain weighted by its corpus size }
        D := 0;
        if Length(Domains) > 1 then
        begin
          Pick := Int64(Rng.NextInt(High(Integer))) mod TotalLen;
          while (D < High(Domains)) and (Pick >= Length(Domains[D])) do
          begin
            Dec(Pick, Length(Domains[D]));
            Inc(D);
          end;
        end;
        Loss := Trainer.TrainStepDomain(Domains[D], Rng, LR, WD,
          D mod Max(1, W.Config.NumExperts), RouterAux);
        if AvgLoss = 0 then
          AvgLoss := Loss
        else
          AvgLoss := 0.95 * AvgLoss + 0.05 * Loss;
        if (S mod 10 = 0) or (S = 1) then
          Say(Format('Step %5d/%d  loss %.4f  (avg %.4f)  %.1f ms/step',
            [S, Steps, Loss, AvgLoss,
             Watch.ElapsedMilliseconds / S]));
        if (SaveEvery > 0) and (S mod SaveEvery = 0) then
        begin
          W.SaveToFile(ModelPath);
          Say('  Checkpoint saved.');
        end;
      end;
      W.SaveToFile(ModelPath);
      Say('Training finished, model saved: ' + ModelPath);
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
      Msgs[0] := TChatMessage.Make('user', ArgValue('prompt', 'Hello!'));
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
      Say(Format('[%d prompt + %d completion tokens]',
        [Usage.PromptTokens, Usage.CompletionTokens]));
    finally
      Gen.Free;
    end;
  finally
    BE.Free; // frees W and Tok
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
      Say('Prism ' + PRISM_VERSION + ' - training CLI');
      Say('Commands:');
      Say('  tokenizer --corpus F --vocab N --out tokenizer.json');
      Say('  tokenize  --corpus F --tokenizer T --out corpus.tokens');
      Say('  init      --tokenizer T --dim N --layers N --heads N --seq N --experts N --out model.prism');
      Say('  train     --model M --tokens F --steps N --batch N --seq N --lr X');
      Say('  sample    --model M --tokenizer T [--chat] --prompt "..."');
    end;
  except
    on E: Exception do
    begin
      Writeln('ERROR: ', E.Message);
      Halt(1);
    end;
  end;
end.
