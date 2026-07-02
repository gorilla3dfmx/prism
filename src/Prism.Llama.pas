unit Prism.Llama;

{ Inferenz-Engine fuer Llama-Architektur-Modelle aus GGUF-Dateien:
  RMSNorm + Rotary Position Embeddings (RoPE) + Grouped-Query-Attention
  + SwiGLU-FFN. Damit laufen existierende trainierte Modelle
  (TinyLlama, Llama 2/3, Mistral, Qwen2, ...) direkt in Prism.

  Speicher-Strategie ("Clustering") wie beim eigenen Format:
  - StreamLayers = 0: alle Layer beim Start in den RAM (schnellste Inferenz)
  - StreamLayers > 0: nur N Layer gleichzeitig im LRU-Cache, der Rest wird
    bedarfsweise aus der GGUF-Datei nachgeladen -> Milliarden-Parameter-
    Modelle auf Geraeten mit wenig RAM (Kosten: Platten-I/O pro Layer).

  Die Gewichte bleiben quantisiert (Q4/Q8) im Speicher und werden in den
  Fused-Kernels aus Prism.Vector direkt verrechnet. }

{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.Classes, System.Math, System.SyncObjs,
  System.Generics.Collections, System.Threading,
  Prism.Types, Prism.Vector, Prism.Gguf;

type
  TLlamaConfig = record
    Dim: Integer;          // embedding_length
    NLayers: Integer;
    NHeads: Integer;
    NKvHeads: Integer;     // GQA; = NHeads bei MHA
    HeadDim: Integer;
    FfnDim: Integer;
    Vocab: Integer;
    CtxLen: Integer;       // effektive Kontextlaenge (ggf. gekappt)
    RopeBase: Single;
    RmsEps: Single;
    RopeNeox: Boolean;
    function QDim: Integer;  // NHeads * HeadDim
    function KvDim: Integer; // NKvHeads * HeadDim
  end;

  TLlamaLayer = class
  public
    AttnNorm, FfnNorm: TArray<Single>;
    Bq, Bk, Bv: TArray<Single>; // optionale Biases (z.B. Qwen2)
    Wq, Wk, Wv, Wo, WGate, WDown, WUp: TQTensor;
  end;

  TLlamaModel = class
  private
    FGg: TGgufFile;
    FStreaming: Boolean;
    FMaxCached: Integer;
    FLayers: TObjectDictionary<Integer, TLlamaLayer>;
    FOrder: TList<Integer>;
    FLock: TCriticalSection;
    FName: string;
    function LoadLayer(L: Integer): TLlamaLayer;
  public
    Cfg: TLlamaConfig;
    TokenEmbd, OutputW: TQTensor;
    OutputNorm: TArray<Single>;
    { CtxOverride > 0 kappt das Kontextfenster (spart KV-Cache-RAM).
      StreamLayers = 0: alles vorladen; > 0: LRU-Streaming mit N Layern. }
    constructor Create(Gg: TGgufFile; CtxOverride: Integer;
      StreamLayers: Integer; const Log: TProc<string>);
    destructor Destroy; override;
    function GetLayer(L: Integer): TLlamaLayer;
    property Name: string read FName;
    property Streaming: Boolean read FStreaming;
    property Gguf: TGgufFile read FGg;
  end;

  TLlamaEngine = class(TLlmEngine)
  private
    FModel: TLlamaModel;
    FPos: Integer;
    FKCache, FVCache: TArray<TArray<Single>>;
    FKCacheCur, FVCacheCur: TArray<Single>; // Cache des aktuellen Layers
    FX, FXb, FQ, FK, FV, FAttOut, FHb, FHb2, FLogits: TArray<Single>;
    FInvFreq: TArray<Single>;
    procedure Rope(Vec: PSingle; NHeadsVec, Pos: Integer);
    procedure Attention;
  public
    constructor Create(AModel: TLlamaModel);
    procedure Reset; override;
    procedure Step(Token: Integer; NeedLogits: Boolean); override;
    function Logits: TArray<Single>; override;
    function VocabSize: Integer; override;
    function MaxContext: Integer; override;
    function Position: Integer; override;
  end;

  TLlamaBackend = class(TLlmBackend)
  private
    FModel: TLlamaModel;
    FTok: TGgufTokenizerBase;
    FTemplate: TChatTemplate;
  public
    constructor Create(const Path: string; CtxOverride: Integer;
      StreamLayers: Integer; const Log: TProc<string>);
    destructor Destroy; override;
    function CreateEngine: TLlmEngine; override;
    function Tokenizer: TLlmTokenizerBase; override;
    function ModelName: string; override;
    function DefaultTemplate: TChatTemplate; override;
    property Template: TChatTemplate read FTemplate write FTemplate;
    property Model: TLlamaModel read FModel;
  end;

implementation

{ TLlamaConfig }

function TLlamaConfig.QDim: Integer;
begin
  Result := NHeads * HeadDim;
end;

function TLlamaConfig.KvDim: Integer;
begin
  Result := NKvHeads * HeadDim;
end;

{ TLlamaModel }

constructor TLlamaModel.Create(Gg: TGgufFile; CtxOverride: Integer;
  StreamLayers: Integer; const Log: TProc<string>);
var
  A: string;
  L: Integer;
  Lay: TLlamaLayer;

  procedure LogMsg(const S: string);
  begin
    if Assigned(Log) then
      Log(S);
  end;

begin
  inherited Create;
  FGg := Gg;
  FLock := TCriticalSection.Create;
  FLayers := TObjectDictionary<Integer, TLlamaLayer>.Create([doOwnsValues]);
  FOrder := TList<Integer>.Create;
  A := Gg.Arch;

  Cfg.Dim := Integer(Gg.MetaInt(A + '.embedding_length', 0));
  Cfg.NLayers := Integer(Gg.MetaInt(A + '.block_count', 0));
  Cfg.NHeads := Integer(Gg.MetaInt(A + '.attention.head_count', 0));
  Cfg.NKvHeads := Integer(Gg.MetaInt(A + '.attention.head_count_kv',
    Cfg.NHeads));
  Cfg.FfnDim := Integer(Gg.MetaInt(A + '.feed_forward_length', 0));
  Cfg.CtxLen := Integer(Gg.MetaInt(A + '.context_length', 2048));
  Cfg.HeadDim := Integer(Gg.MetaInt(A + '.attention.key_length', 0));
  if (Cfg.Dim = 0) or (Cfg.NLayers = 0) or (Cfg.NHeads = 0) then
    raise Exception.CreateFmt(
      'GGUF: Architektur "%s" liefert keine vollstaendige Llama-Konfiguration.',
      [A]);
  if Cfg.HeadDim = 0 then
    Cfg.HeadDim := Cfg.Dim div Cfg.NHeads;
  Cfg.RopeBase := Gg.MetaFloat(A + '.rope.freq_base', 10000.0);
  Cfg.RmsEps := Gg.MetaFloat(A + '.attention.layer_norm_rms_epsilon', 1e-5);
  Cfg.RopeNeox := SameText(A, 'qwen2') or SameText(A, 'qwen2moe') or
    SameText(A, 'qwen3') or SameText(A, 'phi2') or SameText(A, 'phi3') or
    SameText(A, 'stablelm') or SameText(A, 'gptneox') or SameText(A, 'gemma');
  if (CtxOverride > 0) and (CtxOverride < Cfg.CtxLen) then
    Cfg.CtxLen := CtxOverride;

  TokenEmbd := Gg.LoadTensor('token_embd.weight');
  Cfg.Vocab := TokenEmbd.Rows;
  OutputNorm := Gg.LoadTensorF32('output_norm.weight');
  if Gg.HasTensor('output.weight') then
    OutputW := Gg.LoadTensor('output.weight')
  else
    OutputW := TokenEmbd; // Weight-Tying

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
        LogMsg(Format('Layer %d/%d geladen', [L + 1, Cfg.NLayers]));
    end;
  end
  else
    LogMsg(Format('Streaming-Modus: max. %d von %d Layern im RAM',
      [FMaxCached, Cfg.NLayers]));
end;

destructor TLlamaModel.Destroy;
begin
  FLayers.Free;
  FOrder.Free;
  FLock.Free;
  FGg.Free;
  inherited;
end;

function TLlamaModel.LoadLayer(L: Integer): TLlamaLayer;
var
  P: string;
begin
  Result := TLlamaLayer.Create;
  P := Format('blk.%d.', [L]);
  Result.AttnNorm := FGg.LoadTensorF32(P + 'attn_norm.weight');
  Result.FfnNorm := FGg.LoadTensorF32(P + 'ffn_norm.weight');
  Result.Wq := FGg.LoadTensor(P + 'attn_q.weight');
  Result.Wk := FGg.LoadTensor(P + 'attn_k.weight');
  Result.Wv := FGg.LoadTensor(P + 'attn_v.weight');
  Result.Wo := FGg.LoadTensor(P + 'attn_output.weight');
  Result.WGate := FGg.LoadTensor(P + 'ffn_gate.weight');
  Result.WDown := FGg.LoadTensor(P + 'ffn_down.weight');
  Result.WUp := FGg.LoadTensor(P + 'ffn_up.weight');
  if FGg.HasTensor(P + 'attn_q.bias') then
    Result.Bq := FGg.LoadTensorF32(P + 'attn_q.bias');
  if FGg.HasTensor(P + 'attn_k.bias') then
    Result.Bk := FGg.LoadTensorF32(P + 'attn_k.bias');
  if FGg.HasTensor(P + 'attn_v.bias') then
    Result.Bv := FGg.LoadTensorF32(P + 'attn_v.bias');
end;

function TLlamaModel.GetLayer(L: Integer): TLlamaLayer;
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
      FLayers.Remove(FOrder[0]); // doOwnsValues gibt den Layer frei
      FOrder.Delete(0);
    end;
  finally
    FLock.Leave;
  end;
end;

{ TLlamaEngine }

constructor TLlamaEngine.Create(AModel: TLlamaModel);
var
  I, HD: Integer;
begin
  inherited Create;
  FModel := AModel;
  HD := FModel.Cfg.HeadDim;
  SetLength(FX, FModel.Cfg.Dim);
  SetLength(FXb, Max(FModel.Cfg.Dim, FModel.Cfg.QDim));
  SetLength(FQ, FModel.Cfg.QDim);
  SetLength(FK, FModel.Cfg.KvDim);
  SetLength(FV, FModel.Cfg.KvDim);
  SetLength(FAttOut, FModel.Cfg.QDim);
  SetLength(FHb, FModel.Cfg.FfnDim);
  SetLength(FHb2, FModel.Cfg.FfnDim);
  SetLength(FLogits, FModel.Cfg.Vocab);
  SetLength(FInvFreq, HD div 2);
  for I := 0 to HD div 2 - 1 do
    FInvFreq[I] := Power(FModel.Cfg.RopeBase, -2.0 * I / HD);
  Reset;
end;

procedure TLlamaEngine.Reset;
var
  L: Integer;
begin
  FPos := 0;
  SetLength(FKCache, FModel.Cfg.NLayers);
  SetLength(FVCache, FModel.Cfg.NLayers);
  for L := 0 to FModel.Cfg.NLayers - 1 do
  begin
    SetLength(FKCache[L], Int64(FModel.Cfg.CtxLen) * FModel.Cfg.KvDim);
    SetLength(FVCache[L], Int64(FModel.Cfg.CtxLen) * FModel.Cfg.KvDim);
  end;
end;

function TLlamaEngine.Logits: TArray<Single>;
begin
  Result := FLogits;
end;

function TLlamaEngine.VocabSize: Integer;
begin
  Result := FModel.Cfg.Vocab;
end;

function TLlamaEngine.MaxContext: Integer;
begin
  Result := FModel.Cfg.CtxLen;
end;

function TLlamaEngine.Position: Integer;
begin
  Result := FPos;
end;

procedure TLlamaEngine.Rope(Vec: PSingle; NHeadsVec, Pos: Integer);
var
  H, I, HD, Half: Integer;
  P: PSingle;
  Val, FCos, FSin, V0, V1: Single;
begin
  HD := FModel.Cfg.HeadDim;
  Half := HD div 2;
  for H := 0 to NHeadsVec - 1 do
  begin
    P := Vec + H * HD;
    if FModel.Cfg.RopeNeox then
    begin
      for I := 0 to Half - 1 do
      begin
        Val := Pos * FInvFreq[I];
        FCos := Cos(Val);
        FSin := Sin(Val);
        V0 := P[I];
        V1 := P[I + Half];
        P[I] := V0 * FCos - V1 * FSin;
        P[I + Half] := V0 * FSin + V1 * FCos;
      end;
    end
    else
    begin
      for I := 0 to Half - 1 do
      begin
        Val := Pos * FInvFreq[I];
        FCos := Cos(Val);
        FSin := Sin(Val);
        V0 := P[2 * I];
        V1 := P[2 * I + 1];
        P[2 * I] := V0 * FCos - V1 * FSin;
        P[2 * I + 1] := V0 * FSin + V1 * FCos;
      end;
    end;
  end;
end;

procedure TLlamaEngine.Attention;
var
  NH, HD, KvDim, Group, Pos: Integer;
  Scale: Single;
begin
  NH := FModel.Cfg.NHeads;
  HD := FModel.Cfg.HeadDim;
  KvDim := FModel.Cfg.KvDim;
  Group := NH div FModel.Cfg.NKvHeads;
  Pos := FPos;
  Scale := 1.0 / Sqrt(HD);
  TParallel.&For(0, NH - 1,
    procedure(H: Integer)
    var
      Att: TArray<Single>;
      T2, I, KvOff: Integer;
      S: Single;
      Q, O: PSingle;
    begin
      SetLength(Att, Pos + 1);
      Q := PSingle(@FQ[0]) + H * HD;
      KvOff := (H div Group) * HD;
      for T2 := 0 to Pos do
      begin
        S := 0;
        for I := 0 to HD - 1 do
          S := S + Q[I] * FKCacheCur[Int64(T2) * KvDim + KvOff + I];
        Att[T2] := S * Scale;
      end;
      SoftmaxVec(@Att[0], Pos + 1);
      O := PSingle(@FAttOut[0]) + H * HD;
      for I := 0 to HD - 1 do
        O[I] := 0;
      for T2 := 0 to Pos do
      begin
        S := Att[T2];
        for I := 0 to HD - 1 do
          O[I] := O[I] + S * FVCacheCur[Int64(T2) * KvDim + KvOff + I];
      end;
    end);
end;

procedure TLlamaEngine.Step(Token: Integer; NeedLogits: Boolean);
var
  L: Integer;
  Lay: TLlamaLayer;
  C, QD, KvD, Ffn: Integer;
begin
  if FPos >= FModel.Cfg.CtxLen then
    raise Exception.Create('Kontextfenster erschoepft.');
  if (Token < 0) or (Token >= FModel.Cfg.Vocab) then
    raise Exception.CreateFmt('Ungueltiges Token %d', [Token]);
  C := FModel.Cfg.Dim;
  QD := FModel.Cfg.QDim;
  KvD := FModel.Cfg.KvDim;
  Ffn := FModel.Cfg.FfnDim;

  FModel.TokenEmbd.DequantRow(Token, @FX[0]);

  for L := 0 to FModel.Cfg.NLayers - 1 do
  begin
    Lay := FModel.GetLayer(L);
    { Attention }
    RmsNormVec(@FXb[0], @FX[0], @Lay.AttnNorm[0], C, FModel.Cfg.RmsEps);
    Lay.Wq.MatVec(@FQ[0], @FXb[0]);
    Lay.Wk.MatVec(@FK[0], @FXb[0]);
    Lay.Wv.MatVec(@FV[0], @FXb[0]);
    if Length(Lay.Bq) > 0 then
      AddVec(@FQ[0], @Lay.Bq[0], QD);
    if Length(Lay.Bk) > 0 then
      AddVec(@FK[0], @Lay.Bk[0], KvD);
    if Length(Lay.Bv) > 0 then
      AddVec(@FV[0], @Lay.Bv[0], KvD);
    Rope(@FQ[0], FModel.Cfg.NHeads, FPos);
    Rope(@FK[0], FModel.Cfg.NKvHeads, FPos);
    Move(FK[0], FKCache[L][Int64(FPos) * KvD], KvD * SizeOf(Single));
    Move(FV[0], FVCache[L][Int64(FPos) * KvD], KvD * SizeOf(Single));
    FKCacheCur := FKCache[L];
    FVCacheCur := FVCache[L];
    Attention;
    Lay.Wo.MatVec(@FXb[0], @FAttOut[0]);
    AddVec(@FX[0], @FXb[0], C);
    { SwiGLU-FFN }
    RmsNormVec(@FXb[0], @FX[0], @Lay.FfnNorm[0], C, FModel.Cfg.RmsEps);
    Lay.WGate.MatVec(@FHb[0], @FXb[0]);
    Lay.WUp.MatVec(@FHb2[0], @FXb[0]);
    SiluVec(@FHb[0], Ffn);
    MulVec(@FHb[0], @FHb2[0], Ffn);
    Lay.WDown.MatVec(@FXb[0], @FHb[0]);
    AddVec(@FX[0], @FXb[0], C);
  end;

  if NeedLogits then
  begin
    RmsNormVec(@FXb[0], @FX[0], @FModel.OutputNorm[0], C, FModel.Cfg.RmsEps);
    FModel.OutputW.MatVec(@FLogits[0], @FXb[0]);
  end;
  Inc(FPos);
end;

{ TLlamaBackend }

constructor TLlamaBackend.Create(const Path: string; CtxOverride: Integer;
  StreamLayers: Integer; const Log: TProc<string>);
var
  Gg: TGgufFile;
begin
  inherited Create;
  Gg := TGgufFile.Create(Path);
  FModel := TLlamaModel.Create(Gg, CtxOverride, StreamLayers, Log);
  FTok := CreateGgufTokenizer(Gg);
  FTemplate := ctAuto;
end;

destructor TLlamaBackend.Destroy;
begin
  FTok.Free;
  FModel.Free; // gibt auch TGgufFile frei
  inherited;
end;

function TLlamaBackend.CreateEngine: TLlmEngine;
begin
  Result := TLlamaEngine.Create(FModel);
end;

function TLlamaBackend.Tokenizer: TLlmTokenizerBase;
begin
  Result := FTok;
end;

function TLlamaBackend.ModelName: string;
begin
  Result := FModel.Name;
end;

function TLlamaBackend.DefaultTemplate: TChatTemplate;
begin
  if FTemplate <> ctAuto then
    Result := FTemplate
  else
    Result := FTok.AutoTemplate;
end;

end.
