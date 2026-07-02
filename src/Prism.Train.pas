unit Prism.Train;

{ Training of custom Prism models: full forward/backward pass
  (port of the llm.c GPT-2 math) + AdamW optimizer.

  MoE extension ("thematic areas"): when NumExperts > 1, a router
  selects one expert per token (top-1). In the backward pass,
  gradients flow only through the selected expert as well as through
  the gate value into the router (softmax backprop). This way the areas
  specialize on their own during training.

  TTrainingService: background thread for online finetuning via the
  REST interface (samples are appended, N steps trained,
  checkpoint saved). }

{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.Classes, System.Math, System.SyncObjs,
  System.Generics.Collections, System.Threading,
  Prism.Types, Prism.Vector, Prism.Tensor, Prism.Model, Prism.Tokenizer;

type
  TActLayout = record
    B, T, C, L, NH, H, V, E: Integer;
    Encoded, Ln1, Ln1Mean, Ln1Rstd, Qkv, Atty, Preatt, Att, AttProj,
    Residual2, Ln2, Ln2Mean, Ln2Rstd, RouterLogits, RouterProbs, Gate,
    Fch, FchGelu, FcProj, Residual3, Lnf, LnfMean, LnfRstd,
    Logits, Probs, Losses: Int64;
    Total: Int64;
    procedure Init(AB, AT: Integer; const Cfg: TModelConfig);
  end;

  TTrainer = class
  private
    FW: TFullWeights;
    FParams, FGrads, FAdamM, FAdamV: TArray<Single>;
    FLay: TParamLayout;
    FCfg: TModelConfig;
    FB, FT: Integer;
    FActs, FGActs: TArray<Single>;
    AL: TActLayout;
    FExpertIdx: TArray<Integer>; // [L*B*T] selected expert
    FInputs, FTargets: TArray<Integer>;
    FStep: Integer;
    FDomainId: Integer;   // -1 = free routing
    FRouterAux: Single;   // weight of the domain-routing auxiliary loss
    function PP(Off: Int64): PSingle; inline;  // Params
    function PG(Off: Int64): PSingle; inline;  // Grads
    function PA(Off: Int64): PSingle; inline;  // Acts
    function PGA(Off: Int64): PSingle; inline; // Grad-Acts
    procedure MoEForward(L: Integer);
    procedure MoEBackward(L: Integer);
    procedure NextBatch(const Tokens: TArray<Integer>; var Rng: TRng);
  public
    constructor Create(AW: TFullWeights; ABatch, ASeq: Integer);
    function ForwardBackward: Single; // Loss
    procedure Update(LR, Beta1, Beta2, Eps, WeightDecay: Single);
    function TrainStep(const Tokens: TArray<Integer>; var Rng: TRng;
      LR, WeightDecay: Single): Single;
    { Domain-guided MoE training ("thematic areas"): the batch is drawn
      from Tokens and the router additionally receives a cross-entropy
      pull towards expert DomainId (RouterAux = loss weight, e.g. 0.1).
      This is how experts specialize on knowledge domains. }
    function TrainStepDomain(const Tokens: TArray<Integer>; var Rng: TRng;
      LR, WeightDecay: Single; DomainId: Integer;
      RouterAux: Single): Single;
    property StepCount: Integer read FStep;
    property Batch: Integer read FB;
    property SeqLen: Integer read FT;
  end;

  { Background finetuning for the REST server (full-memory mode only) }
  TTrainingService = class(TThread)
  private
    FTrainer: TTrainer;
    FW: TFullWeights;
    FTok: TTokenizer;
    FSharedLock: TCriticalSection; // blocks inference during updates
    FQueue: TThreadedQueue<TBytes>;
    FTokens: TList<Integer>;
    FModelPath: string;
    FRng: TRng;
    FLog: TProc<string>;
  protected
    procedure Execute; override;
  public
    StepsPerJob: Integer;
    LearnRate: Single;
    constructor Create(AW: TFullWeights; ATok: TTokenizer;
      ASharedLock: TCriticalSection; const AModelPath: string;
      const SeedTokens: TArray<Integer>; const ALog: TProc<string>);
    destructor Destroy; override;
    procedure EnqueueSample(const SampleBytes: TBytes);
    procedure Shutdown;
  end;

function LoadTokensFile(const Path: string): TArray<Integer>;
procedure SaveTokensFile(const Path: string; const Tokens: TArray<Integer>);

implementation

function LoadTokensFile(const Path: string): TArray<Integer>;
var
  FS: TFileStream;
  N: Int64;
begin
  FS := TFileStream.Create(Path, fmOpenRead or fmShareDenyWrite);
  try
    N := FS.Size div 4;
    SetLength(Result, N);
    if N > 0 then
      FS.ReadBuffer(Result[0], N * 4);
  finally
    FS.Free;
  end;
end;

procedure SaveTokensFile(const Path: string; const Tokens: TArray<Integer>);
var
  FS: TFileStream;
begin
  FS := TFileStream.Create(Path, fmCreate);
  try
    if Length(Tokens) > 0 then
      FS.WriteBuffer(Tokens[0], Length(Tokens) * 4);
  finally
    FS.Free;
  end;
end;

{ TActLayout }

procedure TActLayout.Init(AB, AT: Integer; const Cfg: TModelConfig);
var
  O, BT, BTC: Int64;
begin
  B := AB;
  T := AT;
  C := Cfg.Dim;
  L := Cfg.NumLayers;
  NH := Cfg.NumHeads;
  H := Cfg.Hidden;
  V := Cfg.VocabSize;
  E := Max(1, Cfg.NumExperts);
  BT := Int64(B) * T;
  BTC := BT * C;
  O := 0;
  Encoded := O; Inc(O, BTC);
  Ln1 := O; Inc(O, L * BTC);
  Ln1Mean := O; Inc(O, L * BT);
  Ln1Rstd := O; Inc(O, L * BT);
  Qkv := O; Inc(O, L * BT * 3 * C);
  Atty := O; Inc(O, L * BTC);
  Preatt := O; Inc(O, Int64(L) * B * NH * T * T);
  Att := O; Inc(O, Int64(L) * B * NH * T * T);
  AttProj := O; Inc(O, L * BTC);
  Residual2 := O; Inc(O, L * BTC);
  Ln2 := O; Inc(O, L * BTC);
  Ln2Mean := O; Inc(O, L * BT);
  Ln2Rstd := O; Inc(O, L * BT);
  RouterLogits := O; if E > 1 then Inc(O, L * BT * E);
  RouterProbs := O; if E > 1 then Inc(O, L * BT * E);
  Gate := O; if E > 1 then Inc(O, L * BT);
  Fch := O; Inc(O, L * BT * H);
  FchGelu := O; Inc(O, L * BT * H);
  FcProj := O; Inc(O, L * BTC);
  Residual3 := O; Inc(O, L * BTC);
  Lnf := O; Inc(O, BTC);
  LnfMean := O; Inc(O, BT);
  LnfRstd := O; Inc(O, BT);
  Logits := O; Inc(O, BT * V);
  Probs := O; Inc(O, BT * V);
  Losses := O; Inc(O, BT);
  Total := O;
end;

{ TTrainer }

constructor TTrainer.Create(AW: TFullWeights; ABatch, ASeq: Integer);
begin
  inherited Create;
  FW := AW;
  FCfg := AW.Config;
  FLay := AW.Layout;
  FParams := AW.Params;
  FB := Max(1, ABatch);
  FT := Min(Max(8, ASeq), FCfg.SeqLen);
  SetLength(FGrads, FLay.TotalCount);
  SetLength(FAdamM, FLay.TotalCount);
  SetLength(FAdamV, FLay.TotalCount);
  AL.Init(FB, FT, FCfg);
  SetLength(FActs, AL.Total);
  SetLength(FGActs, AL.Total);
  SetLength(FExpertIdx, Int64(FCfg.NumLayers) * FB * FT);
  SetLength(FInputs, FB * FT);
  SetLength(FTargets, FB * FT);
  FStep := 0;
  FDomainId := -1;
  FRouterAux := 0;
end;

function TTrainer.PP(Off: Int64): PSingle;
begin
  Result := PSingle(@FParams[0]) + Off;
end;

function TTrainer.PG(Off: Int64): PSingle;
begin
  Result := PSingle(@FGrads[0]) + Off;
end;

function TTrainer.PA(Off: Int64): PSingle;
begin
  Result := PSingle(@FActs[0]) + Off;
end;

function TTrainer.PGA(Off: Int64): PSingle;
begin
  Result := PSingle(@FGActs[0]) + Off;
end;

procedure TTrainer.NextBatch(const Tokens: TArray<Integer>; var Rng: TRng);
var
  BI, I, StartMax, S: Integer;
begin
  StartMax := Length(Tokens) - FT - 1;
  if StartMax < 0 then
    raise Exception.Create('Training data shorter than the sequence length.');
  for BI := 0 to FB - 1 do
  begin
    S := Rng.NextInt(StartMax + 1);
    for I := 0 to FT - 1 do
    begin
      FInputs[BI * FT + I] := Tokens[S + I];
      FTargets[BI * FT + I] := Tokens[S + I + 1];
    end;
  end;
end;

procedure TTrainer.MoEForward(L: Integer);
var
  BT: Int64;
  LB: Int64;
begin
  BT := Int64(FB) * FT;
  LB := FLay.LayerBase(L);
  TParallel.&For(0, Integer(BT) - 1,
    procedure(BTI: Integer)
    var
      X, RL, RP, FchP, GeluP, OP: PSingle;
      EIdx, K: Integer;
      GateV: Single;
      XB: Int64;
    begin
      X := PA(AL.Ln2 + (Int64(L) * BT + BTI) * AL.C);
      RL := PA(AL.RouterLogits + (Int64(L) * BT + BTI) * AL.E);
      RP := PA(AL.RouterProbs + (Int64(L) * BT + BTI) * AL.E);
      MatVecF32(RL, PP(LB + FLay.RRouterW), X, AL.E, AL.C, nil);
      Move(RL^, RP^, AL.E * SizeOf(Single));
      SoftmaxVec(RP, AL.E);
      EIdx := 0;
      for K := 1 to AL.E - 1 do
        if RP[K] > RP[EIdx] then
          EIdx := K;
      GateV := RP[EIdx];
      FExpertIdx[Int64(L) * BT + BTI] := EIdx;
      PA(AL.Gate + Int64(L) * BT)[BTI] := GateV;
      XB := FLay.ExpertBase(L, EIdx);
      FchP := PA(AL.Fch + (Int64(L) * BT + BTI) * AL.H);
      GeluP := PA(AL.FchGelu + (Int64(L) * BT + BTI) * AL.H);
      OP := PA(AL.FcProj + (Int64(L) * BT + BTI) * AL.C);
      MatVecF32(FchP, PP(XB + FLay.XFcW), X, AL.H, AL.C, PP(XB + FLay.XFcB));
      GeluForward(GeluP, FchP, AL.H);
      MatVecF32(OP, PP(XB + FLay.XFc2W), GeluP, AL.C, AL.H,
        PP(XB + FLay.XFc2B));
      ScaledResidualForward(
        PA(AL.Residual3 + (Int64(L) * BT + BTI) * AL.C),
        PA(AL.Residual2 + (Int64(L) * BT + BTI) * AL.C), OP, GateV, AL.C);
    end);
end;

procedure TTrainer.MoEBackward(L: Integer);
var
  BT, LB, XB: Int64;
  BTI: Integer;
  EIdx, I, J, K: Integer;
  GateV, DGate, D, AuxScale: Single;
  DRes3, DRes2, DLn2, X, OP, GeluP, FchP, RP: PSingle;
  TmpDO, TmpDGelu, TmpDFch, TmpDLogit: TArray<Single>;
  WRow: PSingle;
begin
  BT := Int64(FB) * FT;
  LB := FLay.LayerBase(L);
  SetLength(TmpDO, AL.C);
  SetLength(TmpDGelu, AL.H);
  SetLength(TmpDFch, AL.H);
  SetLength(TmpDLogit, AL.E);
  { serial: multiple tokens can hit the same expert ->
    gradient accumulation must not run in parallel }
  for BTI := 0 to Integer(BT) - 1 do
  begin
    EIdx := FExpertIdx[Int64(L) * BT + BTI];
    GateV := PA(AL.Gate + Int64(L) * BT)[BTI];
    XB := FLay.ExpertBase(L, EIdx);
    DRes3 := PGA(AL.Residual3 + (Int64(L) * BT + BTI) * AL.C);
    DRes2 := PGA(AL.Residual2 + (Int64(L) * BT + BTI) * AL.C);
    DLn2 := PGA(AL.Ln2 + (Int64(L) * BT + BTI) * AL.C);
    X := PA(AL.Ln2 + (Int64(L) * BT + BTI) * AL.C);
    OP := PA(AL.FcProj + (Int64(L) * BT + BTI) * AL.C);
    GeluP := PA(AL.FchGelu + (Int64(L) * BT + BTI) * AL.H);
    FchP := PA(AL.Fch + (Int64(L) * BT + BTI) * AL.H);
    RP := PA(AL.RouterProbs + (Int64(L) * BT + BTI) * AL.E);

    { residual3 = residual2 + gate*O }
    DGate := 0;
    for I := 0 to AL.C - 1 do
    begin
      DRes2[I] := DRes2[I] + DRes3[I];
      TmpDO[I] := GateV * DRes3[I];
      DGate := DGate + DRes3[I] * OP[I];
    end;

    { through fc2: dGelu = W2^T * dO; dW2 += dO x Gelu; dB2 += dO }
    FillVec(@TmpDGelu[0], 0, AL.H);
    for I := 0 to AL.C - 1 do
    begin
      D := TmpDO[I];
      if D <> 0 then
      begin
        WRow := PP(XB + FLay.XFc2W + Int64(I) * AL.H);
        for J := 0 to AL.H - 1 do
          TmpDGelu[J] := TmpDGelu[J] + WRow[J] * D;
        WRow := PG(XB + FLay.XFc2W + Int64(I) * AL.H);
        for J := 0 to AL.H - 1 do
          WRow[J] := WRow[J] + GeluP[J] * D;
        PG(XB + FLay.XFc2B)[I] := PG(XB + FLay.XFc2B)[I] + D;
      end;
    end;

    { GELU backward }
    FillVec(@TmpDFch[0], 0, AL.H);
    GeluBackward(@TmpDFch[0], FchP, @TmpDGelu[0], AL.H);

    { through fc1: dLn2 += W1^T * dFch; dW1 += dFch x X; dB1 += dFch }
    for J := 0 to AL.H - 1 do
    begin
      D := TmpDFch[J];
      if D <> 0 then
      begin
        WRow := PP(XB + FLay.XFcW + Int64(J) * AL.C);
        for I := 0 to AL.C - 1 do
          DLn2[I] := DLn2[I] + WRow[I] * D;
        WRow := PG(XB + FLay.XFcW + Int64(J) * AL.C);
        for I := 0 to AL.C - 1 do
          WRow[I] := WRow[I] + X[I] * D;
        PG(XB + FLay.XFcB)[J] := PG(XB + FLay.XFcB)[J] + D;
      end;
    end;

    { Router: gate = softmax(logits)[e] -> softmax backprop only via dGate }
    for K := 0 to AL.E - 1 do
    begin
      if K = EIdx then
        TmpDLogit[K] := GateV * (1.0 - RP[K]) * DGate
      else
        TmpDLogit[K] := -GateV * RP[K] * DGate;
    end;
    { Domain-guided routing: auxiliary cross-entropy that pulls the
      router towards the sample's domain expert ("thematic area") }
    if (FDomainId >= 0) and (FDomainId < AL.E) and (FRouterAux > 0) then
    begin
      AuxScale := FRouterAux / BT;
      for K := 0 to AL.E - 1 do
        if K = FDomainId then
          TmpDLogit[K] := TmpDLogit[K] + AuxScale * (RP[K] - 1.0)
        else
          TmpDLogit[K] := TmpDLogit[K] + AuxScale * RP[K];
    end;
    for K := 0 to AL.E - 1 do
    begin
      D := TmpDLogit[K];
      if D <> 0 then
      begin
        WRow := PP(LB + FLay.RRouterW + Int64(K) * AL.C);
        for I := 0 to AL.C - 1 do
          DLn2[I] := DLn2[I] + WRow[I] * D;
        WRow := PG(LB + FLay.RRouterW + Int64(K) * AL.C);
        for I := 0 to AL.C - 1 do
          WRow[I] := WRow[I] + X[I] * D;
      end;
    end;
  end;
end;

function TTrainer.ForwardBackward: Single;
var
  L: Integer;
  BT, BTC, LB: Int64;
  ResA, DResA: PSingle;
  I: Int64;
  LossSum: Double;
  DLoss: Single;
begin
  BT := Int64(FB) * FT;
  BTC := BT * AL.C;

  { ---------- FORWARD ---------- }
  EncoderForward(PA(AL.Encoded), @FInputs[0], PP(FLay.OffWte),
    PP(FLay.OffWpe), FB, FT, AL.C);
  for L := 0 to AL.L - 1 do
  begin
    LB := FLay.LayerBase(L);
    if L = 0 then
      ResA := PA(AL.Encoded)
    else
      ResA := PA(AL.Residual3 + Int64(L - 1) * BTC);
    LayerNormForward(PA(AL.Ln1 + L * BTC), PA(AL.Ln1Mean + L * BT),
      PA(AL.Ln1Rstd + L * BT), ResA, PP(LB + FLay.RLn1W),
      PP(LB + FLay.RLn1B), FB, FT, AL.C);
    MatMulForward(PA(AL.Qkv + L * BT * 3 * AL.C), PA(AL.Ln1 + L * BTC),
      PP(LB + FLay.RQkvW), PP(LB + FLay.RQkvB), FB, FT, AL.C, 3 * AL.C);
    AttentionForward(PA(AL.Atty + L * BTC),
      PA(AL.Preatt + Int64(L) * FB * AL.NH * FT * FT),
      PA(AL.Att + Int64(L) * FB * AL.NH * FT * FT),
      PA(AL.Qkv + L * BT * 3 * AL.C), FB, FT, AL.C, AL.NH);
    MatMulForward(PA(AL.AttProj + L * BTC), PA(AL.Atty + L * BTC),
      PP(LB + FLay.RProjW), PP(LB + FLay.RProjB), FB, FT, AL.C, AL.C);
    ResidualForward(PA(AL.Residual2 + L * BTC), ResA,
      PA(AL.AttProj + L * BTC), BTC);
    LayerNormForward(PA(AL.Ln2 + L * BTC), PA(AL.Ln2Mean + L * BT),
      PA(AL.Ln2Rstd + L * BT), PA(AL.Residual2 + L * BTC),
      PP(LB + FLay.RLn2W), PP(LB + FLay.RLn2B), FB, FT, AL.C);
    if AL.E > 1 then
      MoEForward(L)
    else
    begin
      MatMulForward(PA(AL.Fch + L * BT * AL.H), PA(AL.Ln2 + L * BTC),
        PP(FLay.ExpertBase(L, 0) + FLay.XFcW),
        PP(FLay.ExpertBase(L, 0) + FLay.XFcB), FB, FT, AL.C, AL.H);
      GeluForward(PA(AL.FchGelu + L * BT * AL.H), PA(AL.Fch + L * BT * AL.H),
        BT * AL.H);
      MatMulForward(PA(AL.FcProj + L * BTC), PA(AL.FchGelu + L * BT * AL.H),
        PP(FLay.ExpertBase(L, 0) + FLay.XFc2W),
        PP(FLay.ExpertBase(L, 0) + FLay.XFc2B), FB, FT, AL.H, AL.C);
      ResidualForward(PA(AL.Residual3 + L * BTC), PA(AL.Residual2 + L * BTC),
        PA(AL.FcProj + L * BTC), BTC);
    end;
  end;
  LayerNormForward(PA(AL.Lnf), PA(AL.LnfMean), PA(AL.LnfRstd),
    PA(AL.Residual3 + Int64(AL.L - 1) * BTC), PP(FLay.OffLnfW),
    PP(FLay.OffLnfB), FB, FT, AL.C);
  MatMulForward(PA(AL.Logits), PA(AL.Lnf), PP(FLay.OffWte), nil,
    FB, FT, AL.C, AL.V);
  SoftmaxForward(PA(AL.Probs), PA(AL.Logits), FB, FT, AL.V);
  CrossEntropyForward(PA(AL.Losses), PA(AL.Probs), @FTargets[0],
    FB, FT, AL.V);
  LossSum := 0;
  for I := 0 to BT - 1 do
    LossSum := LossSum + PA(AL.Losses)[I];
  Result := LossSum / BT;

  { ---------- BACKWARD ---------- }
  FillChar(FGrads[0], Length(FGrads) * SizeOf(Single), 0);
  FillChar(FGActs[0], Length(FGActs) * SizeOf(Single), 0);
  DLoss := 1.0 / BT;
  for I := 0 to BT - 1 do
    PGA(AL.Losses)[I] := DLoss;
  CrossEntropySoftmaxBackward(PGA(AL.Logits), PGA(AL.Losses), PA(AL.Probs),
    @FTargets[0], FB, FT, AL.V);
  MatMulBackward(PGA(AL.Lnf), PG(FLay.OffWte), nil, PGA(AL.Logits),
    PA(AL.Lnf), PP(FLay.OffWte), FB, FT, AL.C, AL.V);
  LayerNormBackward(PGA(AL.Residual3 + Int64(AL.L - 1) * BTC),
    PG(FLay.OffLnfW), PG(FLay.OffLnfB), PGA(AL.Lnf),
    PA(AL.Residual3 + Int64(AL.L - 1) * BTC), PP(FLay.OffLnfW),
    PA(AL.LnfMean), PA(AL.LnfRstd), FB, FT, AL.C);

  for L := AL.L - 1 downto 0 do
  begin
    LB := FLay.LayerBase(L);
    if L = 0 then
    begin
      ResA := PA(AL.Encoded);
      DResA := PGA(AL.Encoded);
    end
    else
    begin
      ResA := PA(AL.Residual3 + Int64(L - 1) * BTC);
      DResA := PGA(AL.Residual3 + Int64(L - 1) * BTC);
    end;

    if AL.E > 1 then
      MoEBackward(L)
    else
    begin
      ResidualBackward(PGA(AL.Residual2 + L * BTC), PGA(AL.FcProj + L * BTC),
        PGA(AL.Residual3 + L * BTC), BTC);
      MatMulBackward(PGA(AL.FchGelu + L * BT * AL.H),
        PG(FLay.ExpertBase(L, 0) + FLay.XFc2W),
        PG(FLay.ExpertBase(L, 0) + FLay.XFc2B), PGA(AL.FcProj + L * BTC),
        PA(AL.FchGelu + L * BT * AL.H),
        PP(FLay.ExpertBase(L, 0) + FLay.XFc2W), FB, FT, AL.H, AL.C);
      GeluBackward(PGA(AL.Fch + L * BT * AL.H), PA(AL.Fch + L * BT * AL.H),
        PGA(AL.FchGelu + L * BT * AL.H), BT * AL.H);
      MatMulBackward(PGA(AL.Ln2 + L * BTC),
        PG(FLay.ExpertBase(L, 0) + FLay.XFcW),
        PG(FLay.ExpertBase(L, 0) + FLay.XFcB), PGA(AL.Fch + L * BT * AL.H),
        PA(AL.Ln2 + L * BTC), PP(FLay.ExpertBase(L, 0) + FLay.XFcW),
        FB, FT, AL.C, AL.H);
    end;

    LayerNormBackward(PGA(AL.Residual2 + L * BTC), PG(LB + FLay.RLn2W),
      PG(LB + FLay.RLn2B), PGA(AL.Ln2 + L * BTC),
      PA(AL.Residual2 + L * BTC), PP(LB + FLay.RLn2W),
      PA(AL.Ln2Mean + L * BT), PA(AL.Ln2Rstd + L * BT), FB, FT, AL.C);
    ResidualBackward(DResA, PGA(AL.AttProj + L * BTC),
      PGA(AL.Residual2 + L * BTC), BTC);
    MatMulBackward(PGA(AL.Atty + L * BTC), PG(LB + FLay.RProjW),
      PG(LB + FLay.RProjB), PGA(AL.AttProj + L * BTC),
      PA(AL.Atty + L * BTC), PP(LB + FLay.RProjW), FB, FT, AL.C, AL.C);
    AttentionBackward(PGA(AL.Qkv + L * BT * 3 * AL.C),
      PGA(AL.Preatt + Int64(L) * FB * AL.NH * FT * FT),
      PGA(AL.Att + Int64(L) * FB * AL.NH * FT * FT),
      PGA(AL.Atty + L * BTC), PA(AL.Qkv + L * BT * 3 * AL.C),
      PA(AL.Att + Int64(L) * FB * AL.NH * FT * FT), FB, FT, AL.C, AL.NH);
    MatMulBackward(PGA(AL.Ln1 + L * BTC), PG(LB + FLay.RQkvW),
      PG(LB + FLay.RQkvB), PGA(AL.Qkv + L * BT * 3 * AL.C),
      PA(AL.Ln1 + L * BTC), PP(LB + FLay.RQkvW), FB, FT, AL.C, 3 * AL.C);
    LayerNormBackward(DResA, PG(LB + FLay.RLn1W), PG(LB + FLay.RLn1B),
      PGA(AL.Ln1 + L * BTC), ResA, PP(LB + FLay.RLn1W),
      PA(AL.Ln1Mean + L * BT), PA(AL.Ln1Rstd + L * BT), FB, FT, AL.C);
  end;
  EncoderBackward(PG(FLay.OffWte), PG(FLay.OffWpe), PGA(AL.Encoded),
    @FInputs[0], FB, FT, AL.C);
end;

procedure TTrainer.Update(LR, Beta1, Beta2, Eps, WeightDecay: Single);
var
  N: Int64;
  Chunks: Integer;
  BC1, BC2: Double;
const
  CHUNK = 262144;
begin
  Inc(FStep);
  N := FLay.TotalCount;
  BC1 := 1.0 - Power(Beta1, FStep);
  BC2 := 1.0 - Power(Beta2, FStep);
  Chunks := Integer((N + CHUNK - 1) div CHUNK);
  TParallel.&For(0, Chunks - 1,
    procedure(CI: Integer)
    var
      I, Lo, Hi: Int64;
      G, M, V, MHat, VHat: Double;
    begin
      Lo := Int64(CI) * CHUNK;
      Hi := Min(Lo + CHUNK, N) - 1;
      for I := Lo to Hi do
      begin
        G := FGrads[I];
        M := Beta1 * FAdamM[I] + (1.0 - Beta1) * G;
        V := Beta2 * FAdamV[I] + (1.0 - Beta2) * G * G;
        FAdamM[I] := M;
        FAdamV[I] := V;
        MHat := M / BC1;
        VHat := V / BC2;
        FParams[I] := FParams[I] -
          LR * (MHat / (Sqrt(VHat) + Eps) + WeightDecay * FParams[I]);
      end;
    end);
end;

function TTrainer.TrainStep(const Tokens: TArray<Integer>; var Rng: TRng;
  LR, WeightDecay: Single): Single;
begin
  Result := TrainStepDomain(Tokens, Rng, LR, WeightDecay, -1, 0);
end;

function TTrainer.TrainStepDomain(const Tokens: TArray<Integer>;
  var Rng: TRng; LR, WeightDecay: Single; DomainId: Integer;
  RouterAux: Single): Single;
begin
  FDomainId := DomainId;
  FRouterAux := RouterAux;
  NextBatch(Tokens, Rng);
  Result := ForwardBackward;
  Update(LR, 0.9, 0.999, 1e-8, WeightDecay);
  FDomainId := -1;
  FRouterAux := 0;
end;

{ TTrainingService }

constructor TTrainingService.Create(AW: TFullWeights; ATok: TTokenizer;
  ASharedLock: TCriticalSection; const AModelPath: string;
  const SeedTokens: TArray<Integer>; const ALog: TProc<string>);
begin
  inherited Create(False);
  FreeOnTerminate := False;
  FW := AW;
  FTok := ATok;
  FSharedLock := ASharedLock;
  FModelPath := AModelPath;
  FLog := ALog;
  FQueue := TThreadedQueue<TBytes>.Create(256, INFINITE, 250);
  FTokens := TList<Integer>.Create;
  if Length(SeedTokens) > 0 then
    FTokens.AddRange(SeedTokens);
  FTrainer := TTrainer.Create(AW, 2, Min(128, AW.Config.SeqLen));
  FRng.Seed(UInt64(TThread.GetTickCount64));
  StepsPerJob := 20;
  LearnRate := 1e-4;
end;

destructor TTrainingService.Destroy;
begin
  FQueue.Free;
  FTokens.Free;
  FTrainer.Free;
  inherited;
end;

procedure TTrainingService.EnqueueSample(const SampleBytes: TBytes);
begin
  FQueue.PushItem(SampleBytes);
end;

procedure TTrainingService.Shutdown;
begin
  Terminate;
  FQueue.DoShutDown;
  WaitFor;
end;

procedure TTrainingService.Execute;
var
  Sample: TBytes;
  NewTokens: TArray<Integer>;
  S: Integer;
  Loss: Single;
  MinLen: Integer;
begin
  while not Terminated do
  begin
    if FQueue.PopItem(Sample) <> wrSignaled then
      Continue;
    if Terminated then
      Break;
    try
      NewTokens := FTok.EncodeData(Sample);
      FTokens.AddRange(NewTokens);
      MinLen := FTrainer.SeqLen + 2;
      while (FTokens.Count > 0) and (FTokens.Count < MinLen) do
        FTokens.AddRange(NewTokens); // pad out a small sample
      if FTokens.Count < MinLen then
        Continue;
      Loss := 0;
      for S := 1 to StepsPerJob do
      begin
        FSharedLock.Enter; // inference is paused during the update
        try
          Loss := FTrainer.TrainStep(FTokens.ToArray, FRng, LearnRate, 0.0);
        finally
          FSharedLock.Leave;
        end;
        if Terminated then
          Break;
      end;
      FSharedLock.Enter;
      try
        FW.SaveToFile(FModelPath);
      finally
        FSharedLock.Leave;
      end;
      if Assigned(FLog) then
        FLog(Format('Online training: %d steps, loss %.4f, corpus %d tokens',
          [StepsPerJob, Loss, FTokens.Count]));
    except
      on E: Exception do
        if Assigned(FLog) then
          FLog('Online training error: ' + E.Message);
    end;
  end;
end;

end.
