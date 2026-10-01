unit Prism.Gemma4;

{ Inference engine for Gemma 4 GGUF models (general.architecture = gemma4),
  e.g. gemma-4-E2B-it. Written against llama.cpp src/models/gemma4.cpp; the
  differences to the plain Llama block in Prism.Llama are large enough that
  folding them in there would have made both harder to read:

  - two kinds of layers, interleaved by attention.sliding_window_pattern:
    sliding-window (window 512, head_dim 256, RoPE base 10k) and global
    (head_dim 512, RoPE base 1M, "proportional" RoPE via rope_freqs)
  - KV sharing: the last attention.shared_kv_layers layers own no K/V
    projection and attend over the cache of the last earlier layer of the
    same kind
  - RMS norm on Q and K per head (weighted), on V per head (unweighted),
    attention scale 1.0 instead of 1/sqrt(head_dim)
  - "sandwich" norms: post-attention and post-FFN norm before each residual
  - GeGLU FFN (tanh-GELU), FFN width per layer
  - per-layer embeddings (E2B/E4B): a second, small embedding per layer and
    token, gated into the residual stream after the FFN
  - a learned scalar per layer output, embedding scaled by sqrt(dim),
    final logit soft-capping

  Every weight goes through TQTensor.MatVec, so the Vulkan backend picks it
  up exactly as for Llama models. Streaming (StreamLayers > 0) works the same
  way too; shared-KV layers only need the source layer's CACHE, never its
  weights, so eviction does not interfere. }

{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.Classes, System.Math, System.SyncObjs,
  System.Generics.Collections, System.Threading,
  Prism.Types, Prism.Vector, Prism.Gguf, Prism.Vulkan;

type
  TGemma4LayerCfg = record
    IsSwa: Boolean;
    NHeads, NKvHeads, HeadDim, NRot, FfnDim: Integer;
    { Layer whose KV cache this layer reads. = own index when the layer has
      its own K/V projection, an earlier layer for shared-KV layers. }
    KvSource: Integer;
    function QDim: Integer;
    function KvDim: Integer;
    function HasKv(L: Integer): Boolean;
  end;

  TGemma4Config = record
    Dim, NLayers, Vocab, CtxLen: Integer;
    RmsEps, Softcap: Single;
    SlidingWindow: Integer;
    PleDim: Integer;          // embedding_length_per_layer_input, 0 = none
    RopeBase, RopeBaseSwa: Single;
    Layers: TArray<TGemma4LayerCfg>;
  end;

  TGemma4Layer = class
  public
    AttnNorm, QNorm, KNorm, PostAttnNorm: TArray<Single>;
    FfnNorm, PostFfwNorm, PostNorm: TArray<Single>;
    Wq, Wk, Wv, Wo, WGate, WUp, WDown, InpGate, Proj: TQTensor;
    OutScale: Single;
    destructor Destroy; override;
  end;

  TGemma4Model = class
  private
    FGg: TGgufFile;
    FStreaming: Boolean;
    FMaxCached: Integer;
    FLayers: TObjectDictionary<Integer, TGemma4Layer>;
    FOrder: TList<Integer>;
    FLock: TCriticalSection;
    FName: string;
    function LoadLayer(L: Integer): TGemma4Layer;
  public
    Cfg: TGemma4Config;
    TokenEmbd, OutputW: TQTensor;
    OutputNorm: TArray<Single>;
    { Per-layer embeddings (only when Cfg.PleDim > 0) }
    PleTokEmbd, PleModelProj: TQTensor;
    PleProjNorm: TArray<Single>;
    RopeFreqs: TArray<Single>; // divisors for the global layers' RoPE
    constructor Create(Gg: TGgufFile; CtxOverride: Integer;
      StreamLayers: Integer; const Log: TProc<string>);
    destructor Destroy; override;
    function GetLayer(L: Integer): TGemma4Layer;
    property Name: string read FName;
    property Gguf: TGgufFile read FGg;
  end;

  TGemma4Engine = class(TLlmEngine)
  private
    FModel: TGemma4Model;
    FPos: Integer;
    FKCache, FVCache: TArray<TArray<Single>>; // only layers with own K/V
    FX, FXb, FQ, FK, FV, FAttOut, FHb, FHb2, FLogits: TArray<Single>;
    FPle, FPleProj, FGate, FOnes: TArray<Single>;
    FInvFreqSwa, FInvFreqFull: TArray<Single>;
    FCosSwa, FSinSwa, FCosFull, FSinFull: TArray<Single>;
    FAttScratch: TArray<TArray<Single>>;
    procedure PrepareRope;
    procedure Rope(Vec: PSingle; NHeadsVec: Integer;
      const LC: TGemma4LayerCfg);
    procedure HeadNorm(Vec: PSingle; NHeadsVec, HD: Integer; W: PSingle);
    procedure Attention(const LC: TGemma4LayerCfg);
    procedure PerLayerInputs(Token: Integer);
  public
    constructor Create(AModel: TGemma4Model);
    procedure Reset; override;
    procedure Step(Token: Integer; NeedLogits: Boolean); override;
    function Logits: TArray<Single>; override;
    function VocabSize: Integer; override;
    function MaxContext: Integer; override;
    function Position: Integer; override;
  end;

  TGemma4Backend = class(TGgufBackend)
  private
    FModel: TGemma4Model;
  protected
    function GetArch: string; override;
    function GetVocab: Integer; override;
    function GetContextLength: Integer; override;
  public
    { Takes ownership of Gg. }
    constructor Create(Gg: TGgufFile; CtxOverride: Integer;
      StreamLayers: Integer; const Log: TProc<string>);
    destructor Destroy; override;
    function CreateEngine: TLlmEngine; override;
    function ModelName: string; override;
    property Model: TGemma4Model read FModel;
  end;

implementation

const
  { Gemma 4 advertises 128k-256k context. Allocating the KV cache for that
    up front costs several GB, so without --ctx we stop at this value. }
  DEFAULT_CTX_CAP = 8192;

{ TGemma4LayerCfg }

function TGemma4LayerCfg.QDim: Integer;
begin
  Result := NHeads * HeadDim;
end;

function TGemma4LayerCfg.KvDim: Integer;
begin
  Result := NKvHeads * HeadDim;
end;

function TGemma4LayerCfg.HasKv(L: Integer): Boolean;
begin
  Result := KvSource = L;
end;

{ TGemma4Layer }

destructor TGemma4Layer.Destroy;
begin
  VulkanEvict(Wq);
  VulkanEvict(Wk);
  VulkanEvict(Wv);
  VulkanEvict(Wo);
  VulkanEvict(WGate);
  VulkanEvict(WUp);
  VulkanEvict(WDown);
  VulkanEvict(InpGate);
  VulkanEvict(Proj);
  inherited;
end;

{ TGemma4Model }

constructor TGemma4Model.Create(Gg: TGgufFile; CtxOverride: Integer;
  StreamLayers: Integer; const Log: TProc<string>);
var
  A: string;
  L, J, NShared, FirstShared: Integer;
  Swa: TArray<Int64>;
  LC: TGemma4LayerCfg;
  Lay: TGemma4Layer;

  procedure LogMsg(const S: string);
  begin
    if Assigned(Log) then
      Log(S);
  end;

  { Several Gemma 4 keys are a scalar in one model and a per-layer array in
    the next (E2B: feed_forward_length, 12B: head_count_kv). }
  function LayerInt(const Key: string; Idx: Integer; Def: Int64): Integer;
  var
    Arr: TArray<Int64>;
  begin
    Arr := Gg.MetaIntArr(Key);
    if Idx < Length(Arr) then
      Result := Integer(Arr[Idx])
    else
      Result := Integer(Gg.MetaInt(Key, Def));
  end;

begin
  inherited Create;
  FGg := Gg;
  FLock := TCriticalSection.Create;
  FLayers := TObjectDictionary<Integer, TGemma4Layer>.Create([doOwnsValues]);
  FOrder := TList<Integer>.Create;
  A := Gg.Arch;

  Cfg.Dim := Integer(Gg.MetaInt(A + '.embedding_length', 0));
  Cfg.NLayers := Integer(Gg.MetaInt(A + '.block_count', 0));
  if (Cfg.Dim = 0) or (Cfg.NLayers = 0) then
    raise Exception.Create('GGUF: incomplete Gemma 4 configuration.');
  Cfg.CtxLen := Integer(Gg.MetaInt(A + '.context_length', 8192));
  Cfg.RmsEps := Gg.MetaFloat(A + '.attention.layer_norm_rms_epsilon', 1e-6);
  Cfg.Softcap := Gg.MetaFloat(A + '.final_logit_softcapping', 0);
  Cfg.SlidingWindow := Integer(Gg.MetaInt(A + '.attention.sliding_window', 512));
  Cfg.PleDim := Integer(Gg.MetaInt(A + '.embedding_length_per_layer_input', 0));
  Cfg.RopeBase := Gg.MetaFloat(A + '.rope.freq_base', 1000000.0);
  Cfg.RopeBaseSwa := Gg.MetaFloat(A + '.rope.freq_base_swa', 10000.0);
  if CtxOverride > 0 then
  begin
    if CtxOverride < Cfg.CtxLen then
      Cfg.CtxLen := CtxOverride;
  end
  else if Cfg.CtxLen > DEFAULT_CTX_CAP then
  begin
    LogMsg(Format('Context capped at %d of %d tokens (raise with --ctx)',
      [DEFAULT_CTX_CAP, Cfg.CtxLen]));
    Cfg.CtxLen := DEFAULT_CTX_CAP;
  end;

  Swa := Gg.MetaIntArr(A + '.attention.sliding_window_pattern');
  NShared := Integer(Gg.MetaInt(A + '.attention.shared_kv_layers', 0));
  FirstShared := Cfg.NLayers - NShared;
  SetLength(Cfg.Layers, Cfg.NLayers);
  for L := 0 to Cfg.NLayers - 1 do
  begin
    LC.IsSwa := (L < Length(Swa)) and (Swa[L] <> 0);
    LC.NHeads := LayerInt(A + '.attention.head_count', L, 0);
    LC.NKvHeads := LayerInt(A + '.attention.head_count_kv', L, LC.NHeads);
    LC.FfnDim := LayerInt(A + '.feed_forward_length', L, 0);
    if LC.IsSwa then
    begin
      LC.HeadDim := Integer(Gg.MetaInt(A + '.attention.key_length_swa', 256));
      LC.NRot := Integer(Gg.MetaInt(A + '.rope.dimension_count_swa', LC.HeadDim));
    end
    else
    begin
      LC.HeadDim := Integer(Gg.MetaInt(A + '.attention.key_length', 512));
      LC.NRot := Integer(Gg.MetaInt(A + '.rope.dimension_count', LC.HeadDim));
    end;
    if (LC.NHeads = 0) or (LC.NKvHeads = 0) or (LC.FfnDim = 0) then
      raise Exception.CreateFmt('GGUF: incomplete Gemma 4 layer %d.', [L]);
    { Shared-KV layer: reuse the last non-shared layer of the same kind.
      (llama.cpp hard-codes FirstShared-2 / -1, which is the same thing for
      every released Gemma 4 layout.) }
    LC.KvSource := L;
    if L >= FirstShared then
    begin
      LC.KvSource := -1;
      for J := FirstShared - 1 downto 0 do
        if Cfg.Layers[J].IsSwa = LC.IsSwa then
        begin
          LC.KvSource := J;
          Break;
        end;
      if LC.KvSource < 0 then
        raise Exception.CreateFmt('GGUF: no KV source for Gemma 4 layer %d.', [L]);
    end;
    Cfg.Layers[L] := LC;
  end;

  TokenEmbd := Gg.LoadTensor('token_embd.weight');
  Cfg.Vocab := TokenEmbd.Rows;
  OutputNorm := Gg.LoadTensorF32('output_norm.weight');
  if Gg.HasTensor('output.weight') then
    OutputW := Gg.LoadTensor('output.weight')
  else
    OutputW := TokenEmbd; // weight tying
  if Gg.HasTensor('rope_freqs.weight') then
    RopeFreqs := Gg.LoadTensorF32('rope_freqs.weight');
  if Cfg.PleDim > 0 then
  begin
    PleTokEmbd := Gg.LoadTensor('per_layer_token_embd.weight');
    PleModelProj := Gg.LoadTensor('per_layer_model_proj.weight');
    PleProjNorm := Gg.LoadTensorF32('per_layer_proj_norm.weight');
    if (PleTokEmbd.Cols <> Cfg.PleDim * Cfg.NLayers) or
      (PleModelProj.Rows <> Cfg.PleDim * Cfg.NLayers) then
      raise Exception.Create('GGUF: unexpected Gemma 4 per-layer embedding shape.');
  end;

  FName := Gg.MetaStr('general.name', ExtractFileName(Gg.Path));
  FStreaming := StreamLayers > 0;
  FMaxCached := StreamLayers;
  if not FStreaming then
  begin
    for L := 0 to Cfg.NLayers - 1 do
    begin
      Lay := LoadLayer(L);
      FLayers.Add(L, Lay);
      if (L mod 4 = 0) or (L = Cfg.NLayers - 1) then
        LogMsg(Format('Layer %d/%d loaded', [L + 1, Cfg.NLayers]));
    end;
  end
  else
    LogMsg(Format('Streaming mode: max. %d of %d layers in RAM',
      [FMaxCached, Cfg.NLayers]));
  LogMsg(Format('Gemma 4: %d layers (%d with own KV), sliding window %d, ' +
    'per-layer embedding %d', [Cfg.NLayers, FirstShared, Cfg.SlidingWindow,
    Cfg.PleDim]));
end;

destructor TGemma4Model.Destroy;
begin
  FLayers.Free;
  FOrder.Free;
  FLock.Free;
  VulkanEvict(TokenEmbd);
  VulkanEvict(OutputW);
  VulkanEvict(PleModelProj);
  FGg.Free;
  inherited;
end;

function TGemma4Model.LoadLayer(L: Integer): TGemma4Layer;
var
  P: string;
  OutScale: TArray<Single>;
begin
  Result := TGemma4Layer.Create;
  try
    P := Format('blk.%d.', [L]);
    Result.AttnNorm := FGg.LoadTensorF32(P + 'attn_norm.weight');
    Result.Wq := FGg.LoadTensor(P + 'attn_q.weight');
    Result.QNorm := FGg.LoadTensorF32(P + 'attn_q_norm.weight');
    if Cfg.Layers[L].HasKv(L) then
    begin
      Result.Wk := FGg.LoadTensor(P + 'attn_k.weight');
      Result.KNorm := FGg.LoadTensorF32(P + 'attn_k_norm.weight');
      { Without attn_v (12B global layers) V is the raw K projection. }
      if FGg.HasTensor(P + 'attn_v.weight') then
        Result.Wv := FGg.LoadTensor(P + 'attn_v.weight');
    end;
    Result.Wo := FGg.LoadTensor(P + 'attn_output.weight');
    Result.PostAttnNorm := FGg.LoadTensorF32(P + 'post_attention_norm.weight');
    Result.FfnNorm := FGg.LoadTensorF32(P + 'ffn_norm.weight');
    Result.WGate := FGg.LoadTensor(P + 'ffn_gate.weight');
    Result.WUp := FGg.LoadTensor(P + 'ffn_up.weight');
    Result.WDown := FGg.LoadTensor(P + 'ffn_down.weight');
    Result.PostFfwNorm := FGg.LoadTensorF32(P + 'post_ffw_norm.weight');
    if FGg.HasTensor(P + 'ffn_gate_inp.weight') then
      raise Exception.Create(
        'GGUF: Gemma 4 mixture-of-experts layers are not supported yet.');
    if Cfg.PleDim > 0 then
    begin
      Result.InpGate := FGg.LoadTensor(P + 'inp_gate.weight');
      Result.Proj := FGg.LoadTensor(P + 'proj.weight');
      Result.PostNorm := FGg.LoadTensorF32(P + 'post_norm.weight');
    end;
    Result.OutScale := 1.0;
    if FGg.HasTensor(P + 'layer_output_scale.weight') then
    begin
      OutScale := FGg.LoadTensorF32(P + 'layer_output_scale.weight');
      Result.OutScale := OutScale[0];
    end;
  except
    Result.Free;
    raise;
  end;
end;

function TGemma4Model.GetLayer(L: Integer): TGemma4Layer;
begin
  if not FStreaming then
    Exit(FLayers[L]);
  FLock.Enter;
  try
    if FLayers.TryGetValue(L, Result) then
    begin
      FOrder.Remove(L);
      FOrder.Add(L);
      Exit;
    end;
    Result := LoadLayer(L);
    FLayers.Add(L, Result);
    FOrder.Add(L);
    while FOrder.Count > FMaxCached do
    begin
      FLayers.Remove(FOrder[0]); // doOwnsValues frees the layer
      FOrder.Delete(0);
    end;
  finally
    FLock.Leave;
  end;
end;

{ TGemma4Engine }

constructor TGemma4Engine.Create(AModel: TGemma4Model);
var
  I, MaxQ, MaxKv, MaxFfn, MaxHd, MaxHeads, NRotSwa, NRotFull: Integer;
  LC: TGemma4LayerCfg;
  F: Single;
begin
  inherited Create;
  FModel := AModel;
  MaxQ := 0; MaxKv := 0; MaxFfn := 0; MaxHd := 0; MaxHeads := 0;
  NRotSwa := 0; NRotFull := 0;
  for LC in FModel.Cfg.Layers do
  begin
    MaxQ := Max(MaxQ, LC.QDim);
    MaxKv := Max(MaxKv, LC.KvDim);
    MaxFfn := Max(MaxFfn, LC.FfnDim);
    MaxHd := Max(MaxHd, LC.HeadDim);
    MaxHeads := Max(MaxHeads, LC.NHeads);
    if LC.IsSwa then
      NRotSwa := LC.NRot
    else
      NRotFull := LC.NRot;
  end;
  SetLength(FX, FModel.Cfg.Dim);
  SetLength(FXb, Max(FModel.Cfg.Dim, MaxQ));
  SetLength(FQ, MaxQ);
  SetLength(FK, MaxKv);
  SetLength(FV, MaxKv);
  SetLength(FAttOut, MaxQ);
  SetLength(FHb, MaxFfn);
  SetLength(FHb2, MaxFfn);
  SetLength(FLogits, FModel.Cfg.Vocab);
  SetLength(FOnes, MaxHd);
  for I := 0 to MaxHd - 1 do
    FOnes[I] := 1.0;
  if FModel.Cfg.PleDim > 0 then
  begin
    SetLength(FPle, FModel.Cfg.PleDim * FModel.Cfg.NLayers);
    SetLength(FPleProj, FModel.Cfg.PleDim * FModel.Cfg.NLayers);
    SetLength(FGate, FModel.Cfg.PleDim);
  end;

  { NEOX RoPE frequencies. Global layers divide by rope_freqs
    ("proportional" RoPE): 1 for the rotated band, 1e30 -- i.e. no rotation
    at all -- for the rest. }
  SetLength(FInvFreqSwa, NRotSwa div 2);
  for I := 0 to NRotSwa div 2 - 1 do
    FInvFreqSwa[I] := Power(FModel.Cfg.RopeBaseSwa, -2.0 * I / NRotSwa);
  SetLength(FInvFreqFull, NRotFull div 2);
  for I := 0 to NRotFull div 2 - 1 do
  begin
    F := Power(FModel.Cfg.RopeBase, -2.0 * I / NRotFull);
    if I < Length(FModel.RopeFreqs) then
      F := F / FModel.RopeFreqs[I];
    FInvFreqFull[I] := F;
  end;
  SetLength(FCosSwa, Length(FInvFreqSwa));
  SetLength(FSinSwa, Length(FInvFreqSwa));
  SetLength(FCosFull, Length(FInvFreqFull));
  SetLength(FSinFull, Length(FInvFreqFull));

  SetLength(FAttScratch, MaxHeads);
  for I := 0 to MaxHeads - 1 do
    SetLength(FAttScratch[I], FModel.Cfg.CtxLen);
  Reset;
end;

procedure TGemma4Engine.Reset;
var
  L: Integer;
begin
  FPos := 0;
  SetLength(FKCache, FModel.Cfg.NLayers);
  SetLength(FVCache, FModel.Cfg.NLayers);
  for L := 0 to FModel.Cfg.NLayers - 1 do
    if FModel.Cfg.Layers[L].HasKv(L) then
    begin
      SetLength(FKCache[L], Int64(FModel.Cfg.CtxLen) * FModel.Cfg.Layers[L].KvDim);
      SetLength(FVCache[L], Int64(FModel.Cfg.CtxLen) * FModel.Cfg.Layers[L].KvDim);
    end;
end;

function TGemma4Engine.Logits: TArray<Single>;
begin
  Result := FLogits;
end;

function TGemma4Engine.VocabSize: Integer;
begin
  Result := FModel.Cfg.Vocab;
end;

function TGemma4Engine.MaxContext: Integer;
begin
  Result := FModel.Cfg.CtxLen;
end;

function TGemma4Engine.Position: Integer;
begin
  Result := FPos;
end;

procedure TGemma4Engine.PrepareRope;
var
  I: Integer;
  V: Double;
begin
  { Same position for every layer of this step: the 35 x 9 heads share two
    cos/sin tables instead of each calling Sin/Cos itself. }
  for I := 0 to High(FInvFreqSwa) do
  begin
    V := Double(FPos) * FInvFreqSwa[I];
    FCosSwa[I] := Cos(V);
    FSinSwa[I] := Sin(V);
  end;
  for I := 0 to High(FInvFreqFull) do
  begin
    V := Double(FPos) * FInvFreqFull[I];
    FCosFull[I] := Cos(V);
    FSinFull[I] := Sin(V);
  end;
end;

procedure TGemma4Engine.Rope(Vec: PSingle; NHeadsVec: Integer;
  const LC: TGemma4LayerCfg);
var
  H, I, Half: Integer;
  P, C, S: PSingle;
  V0, V1: Single;
begin
  Half := LC.NRot div 2;
  if LC.IsSwa then
  begin
    C := @FCosSwa[0];
    S := @FSinSwa[0];
  end
  else
  begin
    C := @FCosFull[0];
    S := @FSinFull[0];
  end;
  for H := 0 to NHeadsVec - 1 do
  begin
    P := Vec + H * LC.HeadDim;
    for I := 0 to Half - 1 do
    begin
      V0 := P[I];
      V1 := P[I + Half];
      P[I] := V0 * C[I] - V1 * S[I];
      P[I + Half] := V0 * S[I] + V1 * C[I];
    end;
  end;
end;

procedure TGemma4Engine.HeadNorm(Vec: PSingle; NHeadsVec, HD: Integer;
  W: PSingle);
var
  H: Integer;
begin
  for H := 0 to NHeadsVec - 1 do
    RmsNormVec(Vec + H * HD, Vec + H * HD, W, HD, FModel.Cfg.RmsEps);
end;

procedure TGemma4Engine.Attention(const LC: TGemma4LayerCfg);
var
  NH, HD, KvDim, Group, Pos, First: Integer;
  KC, VC: PSingle;
begin
  NH := LC.NHeads;
  HD := LC.HeadDim;
  KvDim := LC.KvDim;
  Group := NH div LC.NKvHeads;
  Pos := FPos;
  { Sliding window as in llama.cpp (LLAMA_SWA_TYPE_STANDARD): a key at p is
    visible while Pos - p < window, i.e. the current token plus window-1. }
  First := 0;
  if LC.IsSwa then
    First := Max(0, Pos - FModel.Cfg.SlidingWindow + 1);
  KC := @FKCache[LC.KvSource][0];
  VC := @FVCache[LC.KvSource][0];
  TParallel.&For(0, NH - 1,
    procedure(H: Integer)
    var
      Att, Q, O, KRow, VRow: PSingle;
      T2, I, KvOff, N: Integer;
      S: Single;
    begin
      Att := @FAttScratch[H][0];
      Q := PSingle(@FQ[0]) + H * HD;
      KvOff := (H div Group) * HD;
      N := Pos - First + 1;
      for T2 := First to Pos do
      begin
        KRow := KC + Int64(T2) * KvDim + KvOff;
        S := 0;
        for I := 0 to HD - 1 do
          S := S + Q[I] * KRow[I];
        Att[T2 - First] := S; // Gemma 4: attention scale 1.0
      end;
      SoftmaxVec(Att, N);
      O := PSingle(@FAttOut[0]) + H * HD;
      for I := 0 to HD - 1 do
        O[I] := 0;
      for T2 := First to Pos do
      begin
        S := Att[T2 - First];
        VRow := VC + Int64(T2) * KvDim + KvOff;
        for I := 0 to HD - 1 do
          O[I] := O[I] + S * VRow[I];
      end;
    end);
end;

procedure TGemma4Engine.PerLayerInputs(Token: Integer);
var
  L, I, PD: Integer;
  TokScale, ProjScale, InScale: Single;
  P, E: PSingle;
begin
  { llama.cpp project_per_layer_inputs():
      ple[l] = (rmsnorm(proj(x) / sqrt(dim))[l] + tok_ple[l] * sqrt(pd)) / sqrt(2)
    where x is the already sqrt(dim)-scaled token embedding. }
  PD := FModel.Cfg.PleDim;
  TokScale := Sqrt(PD);
  ProjScale := 1.0 / Sqrt(FModel.Cfg.Dim);
  InScale := 1.0 / Sqrt(2.0);
  FModel.PleTokEmbd.DequantRow(Token, @FPle[0]);
  FModel.PleModelProj.MatVec(@FPleProj[0], @FX[0]);
  for L := 0 to FModel.Cfg.NLayers - 1 do
  begin
    P := PSingle(@FPleProj[0]) + L * PD;
    E := PSingle(@FPle[0]) + L * PD;
    ScaleVec(P, ProjScale, PD);
    RmsNormVec(P, P, @FModel.PleProjNorm[0], PD, FModel.Cfg.RmsEps);
    for I := 0 to PD - 1 do
      E[I] := (P[I] + E[I] * TokScale) * InScale;
  end;
end;

procedure TGemma4Engine.Step(Token: Integer; NeedLogits: Boolean);
var
  L, I, C, KvD, Ffn: Integer;
  Lay: TGemma4Layer;
  LC: TGemma4LayerCfg;
  Cap: Single;
begin
  if FPos >= FModel.Cfg.CtxLen then
    raise Exception.Create('Context window exhausted.');
  if (Token < 0) or (Token >= FModel.Cfg.Vocab) then
    raise Exception.CreateFmt('Invalid token %d', [Token]);
  C := FModel.Cfg.Dim;

  FModel.TokenEmbd.DequantRow(Token, @FX[0]);
  ScaleVec(@FX[0], Sqrt(C), C);
  if FModel.Cfg.PleDim > 0 then
    PerLayerInputs(Token);
  PrepareRope;

  for L := 0 to FModel.Cfg.NLayers - 1 do
  begin
    LC := FModel.Cfg.Layers[L];
    Lay := FModel.GetLayer(L);
    KvD := LC.KvDim;
    Ffn := LC.FfnDim;

    { Attention }
    RmsNormVec(@FXb[0], @FX[0], @Lay.AttnNorm[0], C, FModel.Cfg.RmsEps);
    Lay.Wq.MatVec(@FQ[0], @FXb[0]);
    HeadNorm(@FQ[0], LC.NHeads, LC.HeadDim, @Lay.QNorm[0]);
    Rope(@FQ[0], LC.NHeads, LC);
    if LC.HasKv(L) then
    begin
      Lay.Wk.MatVec(@FK[0], @FXb[0]);
      if not Lay.Wv.IsEmpty then
        Lay.Wv.MatVec(@FV[0], @FXb[0])
      else
        Move(FK[0], FV[0], KvD * SizeOf(Single));
      HeadNorm(@FK[0], LC.NKvHeads, LC.HeadDim, @Lay.KNorm[0]);
      HeadNorm(@FV[0], LC.NKvHeads, LC.HeadDim, @FOnes[0]);
      Rope(@FK[0], LC.NKvHeads, LC);
      Move(FK[0], FKCache[L][Int64(FPos) * KvD], KvD * SizeOf(Single));
      Move(FV[0], FVCache[L][Int64(FPos) * KvD], KvD * SizeOf(Single));
    end;
    Attention(LC);
    Lay.Wo.MatVec(@FXb[0], @FAttOut[0]);
    RmsNormVec(@FXb[0], @FXb[0], @Lay.PostAttnNorm[0], C, FModel.Cfg.RmsEps);
    AddVec(@FX[0], @FXb[0], C);

    { GeGLU FFN }
    RmsNormVec(@FXb[0], @FX[0], @Lay.FfnNorm[0], C, FModel.Cfg.RmsEps);
    Lay.WGate.MatVec(@FHb[0], @FXb[0]);
    Lay.WUp.MatVec(@FHb2[0], @FXb[0]);
    GeluVec(@FHb[0], Ffn);
    MulVec(@FHb[0], @FHb2[0], Ffn);
    Lay.WDown.MatVec(@FXb[0], @FHb[0]);
    RmsNormVec(@FXb[0], @FXb[0], @Lay.PostFfwNorm[0], C, FModel.Cfg.RmsEps);
    AddVec(@FX[0], @FXb[0], C);

    { Per-layer embedding, gated by the residual stream }
    if FModel.Cfg.PleDim > 0 then
    begin
      Lay.InpGate.MatVec(@FGate[0], @FX[0]);
      GeluVec(@FGate[0], FModel.Cfg.PleDim);
      MulVec(@FGate[0], PSingle(@FPle[0]) + L * FModel.Cfg.PleDim,
        FModel.Cfg.PleDim);
      Lay.Proj.MatVec(@FXb[0], @FGate[0]);
      RmsNormVec(@FXb[0], @FXb[0], @Lay.PostNorm[0], C, FModel.Cfg.RmsEps);
      AddVec(@FX[0], @FXb[0], C);
    end;

    if Lay.OutScale <> 1.0 then
      ScaleVec(@FX[0], Lay.OutScale, C);
  end;

  if NeedLogits then
  begin
    RmsNormVec(@FXb[0], @FX[0], @FModel.OutputNorm[0], C, FModel.Cfg.RmsEps);
    FModel.OutputW.MatVec(@FLogits[0], @FXb[0]);
    Cap := FModel.Cfg.Softcap;
    if Cap > 0 then
      for I := 0 to FModel.Cfg.Vocab - 1 do
        FLogits[I] := Cap * Tanh(FLogits[I] / Cap);
  end;
  Inc(FPos);
end;

{ TGemma4Backend }

constructor TGemma4Backend.Create(Gg: TGgufFile; CtxOverride: Integer;
  StreamLayers: Integer; const Log: TProc<string>);
begin
  inherited Create;
  FModel := TGemma4Model.Create(Gg, CtxOverride, StreamLayers, Log);
  FTok := CreateGgufTokenizer(Gg);
  FTemplate := ctAuto;
end;

destructor TGemma4Backend.Destroy;
begin
  FModel.Free; // also frees the TGgufFile
  inherited;   // frees the tokenizer
end;

function TGemma4Backend.CreateEngine: TLlmEngine;
begin
  Result := TGemma4Engine.Create(FModel);
end;

function TGemma4Backend.ModelName: string;
begin
  Result := FModel.Name;
end;

function TGemma4Backend.GetArch: string;
begin
  Result := FModel.Gguf.Arch;
end;

function TGemma4Backend.GetVocab: Integer;
begin
  Result := FModel.Cfg.Vocab;
end;

function TGemma4Backend.GetContextLength: Integer;
begin
  Result := FModel.Cfg.CtxLen;
end;

end.
