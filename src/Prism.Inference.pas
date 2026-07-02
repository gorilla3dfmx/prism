unit Prism.Inference;

{ Token-by-token inference for native Prism models (+ generic
  generator/sampler for ALL engines, including GGUF/Llama).

  Efficiency measures:
  - KV cache: attention only computes against already cached keys/values
  - prefill without logits: while the prompt is being consumed, the
    expensive vocabulary projection (V x C) is skipped entirely
  - MoE routing: only ONE FFN expert ("area") is computed per token
  - large MatVecs run through the compute backend (CPU threads/GPU) }

{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.Classes, System.Math, System.Generics.Collections,
  System.Generics.Defaults, System.Threading,
  Prism.Types, Prism.Vector, Prism.Model, Prism.Tokenizer, Prism.Gpu;

type
  TSamplingParams = record
    Temperature: Single;
    TopK: Integer;
    TopP: Single;
    MaxTokens: Integer;
    Seed: UInt64;
    class function Default: TSamplingParams; static;
  end;

  TUsage = record
    PromptTokens: Integer;
    CompletionTokens: Integer;
  end;

  TPrismEngine = class(TLlmEngine)
  private
    FProv: TWeightsProvider;
    FPos: Integer;
    FKCache, FVCache: TArray<TArray<Single>>; // [Layer][SeqLen*C]
    FX, FXn, FAttOut, FQkv, FHid, FLogits, FRouter: TArray<Single>;
    FExpertHist: TArray<Int64>; // routing counts per expert ("areas")
    procedure Attention(L: Integer);
    procedure FfnExpert(L, E: Integer; Gate: Single);
  public
    constructor Create(AProv: TWeightsProvider);
    procedure Reset; override;
    procedure Step(Token: Integer; NeedLogits: Boolean); override;
    function Logits: TArray<Single>; override;
    function VocabSize: Integer; override;
    function MaxContext: Integer; override;
    function Position: Integer; override;
    { Which thematic areas answered: router decisions per expert,
      accumulated over all tokens and layers since Reset }
    property ExpertHistogram: TArray<Int64> read FExpertHist;
  end;

  { Generator + verification scoring; works against the abstractions,
    so it handles both Prism AND GGUF models. }
  { Tool handler: receives the expression inside <<calc: ...>> and
    returns the exact result (or an error text) to inject }
  TToolHandler = reference to function(const Call: string): string;

  TGenerator = class
  private
    FBackend: TLlmBackend;
    FEngine: TLlmEngine;
    FRng: TRng;
    function SampleToken(const SP: TSamplingParams): Integer;
    function GenerateCore(const PromptTokens: TArray<Integer>;
      const SP: TSamplingParams; const OnToken: TProc<string>;
      out Usage: TUsage; OutTokens: TList<Integer>;
      const ToolHandler: TToolHandler): string;
  public
    constructor Create(ABackend: TLlmBackend);
    destructor Destroy; override;
    property Engine: TLlmEngine read FEngine;
    { Generates text; OnToken (optional) receives decoded UTF-8 pieces }
    function Generate(const PromptTokens: TArray<Integer>;
      const SP: TSamplingParams; const OnToken: TProc<string>;
      out Usage: TUsage; OutTokens: TList<Integer> = nil): string;
    { Like Generate, but with tool calling: when the model emits
      <<calc: EXPR>>, the handler result is injected as <<result: ...>>
      into the output AND the model context, then generation continues }
    function GenerateWithTools(const PromptTokens: TArray<Integer>;
      const SP: TSamplingParams; const OnToken: TProc<string>;
      out Usage: TUsage; const ToolHandler: TToolHandler): string;
    { Sum of ln P(cont | ctx) - basis for perplexity/critic score }
    function ScoreContinuation(const Ctx, Cont: TArray<Integer>): Double;
    function Perplexity(const Ctx, Cont: TArray<Integer>): Double;
  end;

  { Backend for native .prism models }
  TPrismBackend = class(TLlmBackend)
  private
    FProv: TWeightsProvider;
    FTok: TTokenizer;
    FOwnsProv: Boolean;
    FName: string;
  public
    constructor Create(AProv: TWeightsProvider; ATok: TTokenizer;
      const AName: string; AOwnsProv: Boolean = True);
    destructor Destroy; override;
    function CreateEngine: TLlmEngine; override;
    function Tokenizer: TLlmTokenizerBase; override;
    function ModelName: string; override;
    function DefaultTemplate: TChatTemplate; override;
    property Provider: TWeightsProvider read FProv;
    property PrismTokenizer: TTokenizer read FTok;
  end;

implementation

{ TSamplingParams }

class function TSamplingParams.Default: TSamplingParams;
begin
  Result.Temperature := 0.8;
  Result.TopK := 40;
  Result.TopP := 0.95;
  Result.MaxTokens := 256;
  Result.Seed := 0;
end;

{ TPrismEngine }

constructor TPrismEngine.Create(AProv: TWeightsProvider);
var
  C: Integer;
begin
  inherited Create;
  FProv := AProv;
  C := FProv.Config.Dim;
  SetLength(FX, C);
  SetLength(FXn, C);
  SetLength(FAttOut, C);
  SetLength(FQkv, 3 * C);
  SetLength(FHid, FProv.Config.Hidden);
  SetLength(FLogits, FProv.Config.VocabSize);
  SetLength(FRouter, Max(1, FProv.Config.NumExperts));
  Reset;
end;

procedure TPrismEngine.Reset;
var
  L: Integer;
begin
  FPos := 0;
  SetLength(FKCache, FProv.Config.NumLayers);
  SetLength(FVCache, FProv.Config.NumLayers);
  for L := 0 to FProv.Config.NumLayers - 1 do
  begin
    SetLength(FKCache[L], Int64(FProv.Config.SeqLen) * FProv.Config.Dim);
    SetLength(FVCache[L], Int64(FProv.Config.SeqLen) * FProv.Config.Dim);
  end;
  FExpertHist := nil;
  SetLength(FExpertHist, Max(1, FProv.Config.NumExperts));
end;

function TPrismEngine.Logits: TArray<Single>;
begin
  Result := FLogits;
end;

function TPrismEngine.VocabSize: Integer;
begin
  Result := FProv.Config.VocabSize;
end;

function TPrismEngine.MaxContext: Integer;
begin
  Result := FProv.Config.SeqLen;
end;

function TPrismEngine.Position: Integer;
begin
  Result := FPos;
end;

procedure TPrismEngine.Attention(L: Integer);
var
  C, NH, HS: Integer;
  Scale: Single;
  K, V: TArray<Single>;
  Pos: Integer;
begin
  C := FProv.Config.Dim;
  NH := FProv.Config.NumHeads;
  HS := C div NH;
  Scale := 1.0 / Sqrt(HS);
  K := FKCache[L];
  V := FVCache[L];
  Pos := FPos;
  TParallel.&For(0, NH - 1,
    procedure(H: Integer)
    var
      Att: TArray<Single>;
      T2, I, HOff: Integer;
      S: Single;
      Q, O: PSingle;
    begin
      SetLength(Att, Pos + 1);
      HOff := H * HS;
      Q := PSingle(@FQkv[0]) + HOff;
      for T2 := 0 to Pos do
      begin
        S := 0;
        for I := 0 to HS - 1 do
          S := S + Q[I] * K[Int64(T2) * C + HOff + I];
        Att[T2] := S * Scale;
      end;
      SoftmaxVec(@Att[0], Pos + 1);
      O := PSingle(@FAttOut[0]) + HOff;
      for I := 0 to HS - 1 do
        O[I] := 0;
      for T2 := 0 to Pos do
      begin
        S := Att[T2];
        for I := 0 to HS - 1 do
          O[I] := O[I] + S * V[Int64(T2) * C + HOff + I];
      end;
    end);
end;

procedure TPrismEngine.FfnExpert(L, E: Integer; Gate: Single);
var
  Arr: TArray<Single>;
  Base: Int64;
  C, H, I: Integer;
  Lay: TParamLayout;
begin
  Lay := FProv.Layout;
  C := FProv.Config.Dim;
  H := FProv.Config.Hidden;
  FProv.GetExpert(L, E, Arr, Base);
  Backend.MatVecF32W(@FHid[0], PSingle(@Arr[0]) + Base + Lay.XFcW, @FXn[0],
    H, C, PSingle(@Arr[0]) + Base + Lay.XFcB, Pointer(Arr), Base + Lay.XFcW);
  GeluVec(@FHid[0], H);
  Backend.MatVecF32W(@FXn[0], PSingle(@Arr[0]) + Base + Lay.XFc2W, @FHid[0],
    C, H, PSingle(@Arr[0]) + Base + Lay.XFc2B, Pointer(Arr), Base + Lay.XFc2W);
  for I := 0 to C - 1 do
    FX[I] := FX[I] + Gate * FXn[I];
end;

procedure TPrismEngine.Step(Token: Integer; NeedLogits: Boolean);
var
  L, I, C, E, BestE: Integer;
  Arr: TArray<Single>;
  Base: Int64;
  Lay: TParamLayout;
  Res: TArray<Single>;
  Gate: Single;
begin
  if FPos >= FProv.Config.SeqLen then
    raise Exception.Create('Context window exhausted.');
  if (Token < 0) or (Token >= FProv.Config.VocabSize) then
    raise Exception.CreateFmt('Invalid token %d', [Token]);
  Lay := FProv.Layout;
  C := FProv.Config.Dim;
  Res := FProv.Resident;

  { x = wte[token] + wpe[pos] }
  for I := 0 to C - 1 do
    FX[I] := Res[Lay.OffWte + Int64(Token) * C + I] +
      Res[Lay.OffWpe + Int64(FPos) * C + I];

  for L := 0 to FProv.Config.NumLayers - 1 do
  begin
    FProv.GetLayer(L, Arr, Base);
    { Attention block }
    LayerNormVec(@FXn[0], @FX[0], PSingle(@Arr[0]) + Base + Lay.RLn1W,
      PSingle(@Arr[0]) + Base + Lay.RLn1B, C, 1e-5);
    Backend.MatVecF32W(@FQkv[0], PSingle(@Arr[0]) + Base + Lay.RQkvW,
      @FXn[0], 3 * C, C, PSingle(@Arr[0]) + Base + Lay.RQkvB,
      Pointer(Arr), Base + Lay.RQkvW);
    Move(FQkv[C], FKCache[L][Int64(FPos) * C], C * SizeOf(Single));
    Move(FQkv[2 * C], FVCache[L][Int64(FPos) * C], C * SizeOf(Single));
    Attention(L);
    Backend.MatVecF32W(@FXn[0], PSingle(@Arr[0]) + Base + Lay.RProjW,
      @FAttOut[0], C, C, PSingle(@Arr[0]) + Base + Lay.RProjB,
      Pointer(Arr), Base + Lay.RProjW);
    AddVec(@FX[0], @FXn[0], C);

    { FFN block: dense or mixture-of-experts (top-1 routing) }
    LayerNormVec(@FXn[0], @FX[0], PSingle(@Arr[0]) + Base + Lay.RLn2W,
      PSingle(@Arr[0]) + Base + Lay.RLn2B, C, 1e-5);
    if FProv.Config.IsMoE then
    begin
      MatVecF32(@FRouter[0], PSingle(@Arr[0]) + Base + Lay.RRouterW,
        @FXn[0], FProv.Config.NumExperts, C, nil);
      SoftmaxVec(@FRouter[0], FProv.Config.NumExperts);
      BestE := 0;
      for E := 1 to FProv.Config.NumExperts - 1 do
        if FRouter[E] > FRouter[BestE] then
          BestE := E;
      Gate := FRouter[BestE];
      Inc(FExpertHist[BestE]);
      FfnExpert(L, BestE, Gate);
    end
    else
      FfnExpert(L, 0, 1.0);
  end;

  if NeedLogits then
  begin
    LayerNormVec(@FXn[0], @FX[0], PSingle(@Res[0]) + Lay.OffLnfW,
      PSingle(@Res[0]) + Lay.OffLnfB, C, 1e-5);
    { Logits = wte * x (Weight-Tying) }
    Backend.MatVecF32W(@FLogits[0], PSingle(@Res[0]) + Lay.OffWte, @FXn[0],
      FProv.Config.VocabSize, C, nil, Pointer(Res), Lay.OffWte);
  end;
  Inc(FPos);
end;

{ TGenerator }

constructor TGenerator.Create(ABackend: TLlmBackend);
begin
  inherited Create;
  FBackend := ABackend;
  FEngine := ABackend.CreateEngine;
  FRng.Seed(UInt64(TThread.GetTickCount64) xor UInt64(NativeUInt(Self)));
end;

destructor TGenerator.Destroy;
begin
  FEngine.Free;
  inherited;
end;

function TGenerator.SampleToken(const SP: TSamplingParams): Integer;
var
  V, I, K: Integer;
  Probs: TArray<Single>;
  Idx: TArray<Integer>;
  L: TArray<Single>;
  Cum: Double;
  R: Single;
begin
  L := FEngine.Logits;
  V := FEngine.VocabSize;
  if SP.Temperature <= 0 then
    Exit(ArgMax(@L[0], V));

  SetLength(Probs, V);
  for I := 0 to V - 1 do
    Probs[I] := L[I] / SP.Temperature;
  SoftmaxVec(@Probs[0], V);

  SetLength(Idx, V);
  for I := 0 to V - 1 do
    Idx[I] := I;
  TArray.Sort<Integer>(Idx, TComparer<Integer>.Construct(
    function(const A, B: Integer): Integer
    begin
      if Probs[A] > Probs[B] then
        Result := -1
      else if Probs[A] < Probs[B] then
        Result := 1
      else
        Result := 0;
    end));

  K := V;
  if (SP.TopK > 0) and (SP.TopK < K) then
    K := SP.TopK;
  if (SP.TopP > 0) and (SP.TopP < 1.0) then
  begin
    Cum := 0;
    for I := 0 to K - 1 do
    begin
      Cum := Cum + Probs[Idx[I]];
      if Cum >= SP.TopP then
      begin
        K := I + 1;
        Break;
      end;
    end;
  end;

  Cum := 0;
  for I := 0 to K - 1 do
    Cum := Cum + Probs[Idx[I]];
  R := FRng.NextSingle * Cum;
  Cum := 0;
  Result := Idx[K - 1];
  for I := 0 to K - 1 do
  begin
    Cum := Cum + Probs[Idx[I]];
    if R < Cum then
      Exit(Idx[I]);
  end;
end;

function TGenerator.GenerateCore(const PromptTokens: TArray<Integer>;
  const SP: TSamplingParams; const OnToken: TProc<string>;
  out Usage: TUsage; OutTokens: TList<Integer>;
  const ToolHandler: TToolHandler): string;
const
  TOOL_OPEN = '<<calc:';
  TOOL_CLOSE = '>>';
  MAX_TOOL_CALLS = 8;
var
  Prompt, InjTokens: TArray<Integer>;
  I, N, Tok, MaxCtx, KeepLen, FlushLen: Integer;
  Buf, Piece: TBytes;
  SB: TStringBuilder;
  Chunk, Accum, Expr, Inj: string;
  ScanPos, A, B, ToolCalls: Integer;
  ToolsOn: Boolean;

  procedure Emit(const S: string);
  begin
    SB.Append(S);
    Accum := Accum + S;
    if Assigned(OnToken) then
      OnToken(S);
  end;

begin
  MaxCtx := FEngine.MaxContext;
  Prompt := PromptTokens;
  if Length(Prompt) = 0 then
    Prompt := [FBackend.Tokenizer.BosId];
  { Truncate the prompt on the left if necessary (context window) }
  KeepLen := MaxCtx - Max(16, Min(SP.MaxTokens, MaxCtx div 4));
  if Length(Prompt) > KeepLen then
    Prompt := Copy(Prompt, Length(Prompt) - KeepLen, KeepLen);

  Usage.PromptTokens := Length(Prompt);
  Usage.CompletionTokens := 0;

  FEngine.Reset;
  { Prefill: compute logits only for the last prompt token }
  for I := 0 to High(Prompt) - 1 do
    FEngine.Step(Prompt[I], False);
  FEngine.Step(Prompt[High(Prompt)], True);

  SB := TStringBuilder.Create;
  try
    Buf := nil;
    N := 0;
    Accum := '';
    ScanPos := 1;
    ToolCalls := 0;
    ToolsOn := Assigned(ToolHandler);
    while (N < SP.MaxTokens) and (FEngine.Position < MaxCtx) do
    begin
      Tok := SampleToken(SP);
      if FBackend.Tokenizer.IsStopToken(Tok) then
        Break;
      if OutTokens <> nil then
        OutTokens.Add(Tok);
      Inc(N);
      Piece := FBackend.Tokenizer.TokenBytes(Tok);
      if Length(Piece) > 0 then
      begin
        Buf := Buf + Piece;
        FlushLen := Utf8CompletePrefixLength(Buf);
        if FlushLen > 0 then
        begin
          Chunk := TEncoding.UTF8.GetString(Buf, 0, FlushLen);
          Emit(Chunk);
          Buf := Copy(Buf, FlushLen, Length(Buf) - FlushLen);
        end;
      end;
      if FEngine.Position >= MaxCtx then
        Break;
      FEngine.Step(Tok, True);

      { Tool call: as soon as a complete <<calc: ...>> marker has been
        emitted, evaluate it and inject <<result: ...>> into both the
        output and the model context, then continue generating }
      if ToolsOn then
      begin
        A := Pos(TOOL_OPEN, Accum, ScanPos);
        if A > 0 then
        begin
          B := Pos(TOOL_CLOSE, Accum, A + Length(TOOL_OPEN));
          if B > 0 then
          begin
            Expr := Trim(Copy(Accum, A + Length(TOOL_OPEN),
              B - A - Length(TOOL_OPEN)));
            Inj := ' <<result: ' + ToolHandler(Expr) + '>>'#10;
            Emit(Inj);
            ScanPos := Length(Accum) + 1;
            Inc(ToolCalls);
            if ToolCalls >= MAX_TOOL_CALLS then
              ToolsOn := False;
            InjTokens := FBackend.Tokenizer.Encode(Inj);
            for I := 0 to High(InjTokens) do
            begin
              if FEngine.Position >= MaxCtx then
                Break;
              FEngine.Step(InjTokens[I], I = High(InjTokens));
            end;
            if FEngine.Position >= MaxCtx then
              Break;
          end;
        end;
      end;
    end;
    if Length(Buf) > 0 then
      Emit(BytesToUtf8Lossy(Buf));
    Usage.CompletionTokens := N;
    Result := SB.ToString;
  finally
    SB.Free;
  end;
end;

function TGenerator.Generate(const PromptTokens: TArray<Integer>;
  const SP: TSamplingParams; const OnToken: TProc<string>;
  out Usage: TUsage; OutTokens: TList<Integer>): string;
begin
  Result := GenerateCore(PromptTokens, SP, OnToken, Usage, OutTokens, nil);
end;

function TGenerator.GenerateWithTools(const PromptTokens: TArray<Integer>;
  const SP: TSamplingParams; const OnToken: TProc<string>;
  out Usage: TUsage; const ToolHandler: TToolHandler): string;
begin
  Result := GenerateCore(PromptTokens, SP, OnToken, Usage, nil, ToolHandler);
end;

function TGenerator.ScoreContinuation(const Ctx, Cont: TArray<Integer>): Double;
var
  I, V: Integer;
  Probs: TArray<Single>;
  P: Single;
begin
  Result := 0;
  if (Length(Ctx) = 0) or (Length(Cont) = 0) then
    Exit;
  if Length(Ctx) + Length(Cont) > FEngine.MaxContext then
    Exit(-1e30);
  V := FEngine.VocabSize;
  SetLength(Probs, V);
  FEngine.Reset;
  for I := 0 to High(Ctx) - 1 do
    FEngine.Step(Ctx[I], False);
  FEngine.Step(Ctx[High(Ctx)], True);
  for I := 0 to High(Cont) do
  begin
    Move(FEngine.Logits[0], Probs[0], V * SizeOf(Single));
    SoftmaxVec(@Probs[0], V);
    P := Probs[Cont[I]];
    if P < 1e-12 then
      P := 1e-12;
    Result := Result + Ln(P);
    if I < High(Cont) then
      FEngine.Step(Cont[I], True)
    else
      Break;
  end;
end;

function TGenerator.Perplexity(const Ctx, Cont: TArray<Integer>): Double;
var
  Lp: Double;
begin
  if Length(Cont) = 0 then
    Exit(1);
  Lp := ScoreContinuation(Ctx, Cont);
  Result := Exp(-Lp / Length(Cont));
end;

{ TPrismBackend }

constructor TPrismBackend.Create(AProv: TWeightsProvider; ATok: TTokenizer;
  const AName: string; AOwnsProv: Boolean);
begin
  inherited Create;
  FProv := AProv;
  FTok := ATok;
  FName := AName;
  FOwnsProv := AOwnsProv;
end;

destructor TPrismBackend.Destroy;
begin
  if FOwnsProv then
    FProv.Free;
  FTok.Free;
  inherited;
end;

function TPrismBackend.CreateEngine: TLlmEngine;
begin
  Result := TPrismEngine.Create(FProv);
end;

function TPrismBackend.Tokenizer: TLlmTokenizerBase;
begin
  Result := FTok;
end;

function TPrismBackend.ModelName: string;
begin
  Result := FName;
end;

function TPrismBackend.DefaultTemplate: TChatTemplate;
begin
  Result := ctPrism;
end;

end.
