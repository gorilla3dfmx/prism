unit Prism.Types;

{ Prism - An LLM framework in pure Object Pascal (Delphi 13).
  Base types, configuration, parameter layout, RNG and the abstract
  base classes for engines/tokenizers/backends.

  IMPORTANT: All sizes/offsets are Int64-capable (NativeInt on 64-bit)
  so that models with billions of parameters are addressable.
  For large models always compile for 64-bit targets (Win64, Linux64,
  Android64, iOS/macOS ARM64). }

interface

uses
  System.SysUtils;

const
  PRISM_VERSION = '0.1.0';

  { Custom checkpoint format (trainable models) }
  CHECKPOINT_MAGIC: UInt32 = $4D535250; // 'PRSM' little-endian
  CHECKPOINT_VERSION = 1;
  CHECKPOINT_HEADER_SIZE = 64;

type
  { Configuration of the custom (trainable) GPT-2-style model.
    NumExperts > 1 enables Mixture-of-Experts ("topical areas"):
    a router picks ONE expert per token (top-1) - only this subgraph
    is computed and has to reside in memory. }
  TModelConfig = record
    VocabSize: Integer;
    SeqLen: Integer;      // maximum context length
    Dim: Integer;         // embedding dimension C
    NumLayers: Integer;
    NumHeads: Integer;
    NumExperts: Integer;  // 1 = classic dense FFN
    function Hidden: Integer;   // MLP hidden = 4*C (per expert)
    function HeadSize: Integer; // C div NumHeads
    function IsMoE: Boolean;
    function ToString: string;
  end;

  { Flat layout of all parameters. Order:
    [wte][wpe][lnf_w][lnf_b][layer 0][layer 1]...[layer L-1]
    Layer block: [core: ln1, qkv, proj, ln2, router][expert 0]...[expert E-1]
    The "resident" prefix (embeddings + final norm) always stays in memory
    during streaming; layer cores and individual experts are loaded on
    demand as clusters (LRU). }
  TParamLayout = record
    Config: TModelConfig;
    OffWte, OffWpe, OffLnfW, OffLnfB: Int64;
    ResidentCount: Int64;   // number of floats in the resident prefix
    LayerCoreCount: Int64;  // floats in the layer core (attention + router)
    ExpertCount: Int64;     // floats per expert (FFN)
    LayerCount: Int64;      // Core + NumExperts * ExpertCount
    TotalCount: Int64;
    { Offsets WITHIN the layer core (relative to the layer base) }
    RLn1W, RLn1B, RQkvW, RQkvB, RProjW, RProjB,
    RLn2W, RLn2B, RRouterW: Int64;
    { Offsets WITHIN an expert block (relative to the expert base) }
    XFcW, XFcB, XFc2W, XFc2B: Int64;
    procedure Init(const AConfig: TModelConfig);
    function LayerBase(L: Integer): Int64;
    function ExpertBase(L, E: Integer): Int64;
  end;

  { Deterministic RNG (xorshift64*), independent of the system Random }
  TRng = record
    State: UInt64;
    procedure Seed(ASeed: UInt64);
    function NextUInt32: UInt32;
    function NextSingle: Single;             // [0, 1)
    function NextGauss: Single;              // N(0, 1)
    function NextInt(Range: Integer): Integer; // [0, Range)
  end;

  TChatMessage = record
    Role: string;    // 'system' | 'user' | 'assistant'
    Content: string;
    class function Make(const ARole, AContent: string): TChatMessage; static;
  end;
  TChatMessages = TArray<TChatMessage>;

  TChatTemplate = (ctAuto, ctPrism, ctChatML, ctLlama2, ctPlain);

  { Abstract tokenizer: implemented by TTokenizer (custom byte-level BPE),
    TSpmTokenizer and TGpt2Tokenizer (both built from GGUF metadata). }
  TLlmTokenizerBase = class
  public
    function VocabSize: Integer; virtual; abstract;
    function Encode(const Text: string): TArray<Integer>; virtual; abstract;
    function TokenBytes(Id: Integer): TBytes; virtual; abstract;
    function Decode(const Ids: TArray<Integer>): string; virtual;
    function BosId: Integer; virtual; abstract;
    function EosId: Integer; virtual; abstract;
    function IsStopToken(Id: Integer): Boolean; virtual;
    { True if raw prompts (completion/generate) should be prefixed with a
      BOS token (GGUF models expect this; the Prism corpus format does not) }
    function PrependBos: Boolean; virtual;
    { Builds the prompt token sequence from chat messages (incl. template) }
    function BuildChatTokens(const Msgs: TChatMessages;
      Template: TChatTemplate): TArray<Integer>; virtual; abstract;
  end;

  { Abstract inference engine (one instance per request; holds the KV cache).
    Step processes ONE token; NeedLogits=False during prompt prefill skips
    the expensive logits projection (V x C) - an important performance win. }
  TLlmEngine = class
  public
    procedure Reset; virtual; abstract;
    procedure Step(Token: Integer; NeedLogits: Boolean); virtual; abstract;
    function Logits: TArray<Single>; virtual; abstract;
    function VocabSize: Integer; virtual; abstract;
    function MaxContext: Integer; virtual; abstract;
    function Position: Integer; virtual; abstract;
  end;

  { Backend = loaded model + tokenizer; factory for engines }
  TLlmBackend = class
  public
    function CreateEngine: TLlmEngine; virtual; abstract;
    function Tokenizer: TLlmTokenizerBase; virtual; abstract;
    function ModelName: string; virtual; abstract;
    function DefaultTemplate: TChatTemplate; virtual;
  end;

function BytesToUtf8Lossy(const B: TBytes): string;
function Utf8CompletePrefixLength(const B: TBytes): Integer;

implementation

uses
  System.Math;

{ TModelConfig }

function TModelConfig.Hidden: Integer;
begin
  Result := 4 * Dim;
end;

function TModelConfig.HeadSize: Integer;
begin
  Result := Dim div NumHeads;
end;

function TModelConfig.IsMoE: Boolean;
begin
  Result := NumExperts > 1;
end;

function TModelConfig.ToString: string;
begin
  Result := Format('vocab=%d seq=%d dim=%d layers=%d heads=%d experts=%d',
    [VocabSize, SeqLen, Dim, NumLayers, NumHeads, NumExperts]);
end;

{ TParamLayout }

procedure TParamLayout.Init(const AConfig: TModelConfig);
var
  C, H, E, O: Int64;
begin
  Config := AConfig;
  C := AConfig.Dim;
  H := AConfig.Hidden;
  E := AConfig.NumExperts;
  if E < 1 then
    E := 1;
  OffWte := 0;
  OffWpe := OffWte + Int64(AConfig.VocabSize) * C;
  OffLnfW := OffWpe + Int64(AConfig.SeqLen) * C;
  OffLnfB := OffLnfW + C;
  ResidentCount := OffLnfB + C;
  { Layer core }
  O := 0;
  RLn1W := O; Inc(O, C);
  RLn1B := O; Inc(O, C);
  RQkvW := O; Inc(O, 3 * C * C);
  RQkvB := O; Inc(O, 3 * C);
  RProjW := O; Inc(O, C * C);
  RProjB := O; Inc(O, C);
  RLn2W := O; Inc(O, C);
  RLn2B := O; Inc(O, C);
  RRouterW := O;
  if E > 1 then
    Inc(O, E * C); // router only for MoE
  LayerCoreCount := O;
  { Expert block }
  O := 0;
  XFcW := O; Inc(O, H * C);
  XFcB := O; Inc(O, H);
  XFc2W := O; Inc(O, C * H);
  XFc2B := O; Inc(O, C);
  ExpertCount := O;
  LayerCount := LayerCoreCount + E * ExpertCount;
  TotalCount := ResidentCount + Int64(AConfig.NumLayers) * LayerCount;
end;

function TParamLayout.LayerBase(L: Integer): Int64;
begin
  Result := ResidentCount + Int64(L) * LayerCount;
end;

function TParamLayout.ExpertBase(L, E: Integer): Int64;
begin
  Result := LayerBase(L) + LayerCoreCount + Int64(E) * ExpertCount;
end;

{ TRng }

procedure TRng.Seed(ASeed: UInt64);
begin
  if ASeed = 0 then
    ASeed := $9E3779B97F4A7C15;
  State := ASeed;
end;

function TRng.NextUInt32: UInt32;
begin
  State := State xor (State shl 13);
  State := State xor (State shr 7);
  State := State xor (State shl 17);
  Result := UInt32(State shr 32);
end;

function TRng.NextSingle: Single;
begin
  Result := (NextUInt32 shr 8) * (1.0 / 16777216.0);
end;

function TRng.NextGauss: Single;
var
  U1, U2: Double;
begin
  U1 := NextSingle;
  if U1 < 1e-12 then
    U1 := 1e-12;
  U2 := NextSingle;
  Result := Sqrt(-2.0 * Ln(U1)) * Cos(2.0 * Pi * U2);
end;

function TRng.NextInt(Range: Integer): Integer;
begin
  if Range <= 0 then
    Exit(0);
  Result := Integer(NextUInt32 mod UInt32(Range));
end;

{ TChatMessage }

class function TChatMessage.Make(const ARole, AContent: string): TChatMessage;
begin
  Result.Role := ARole;
  Result.Content := AContent;
end;

{ TLlmTokenizerBase }

function TLlmTokenizerBase.Decode(const Ids: TArray<Integer>): string;
var
  Buf: TBytes;
  I: Integer;
  Piece: TBytes;
  N: Integer;
begin
  Buf := nil;
  N := 0;
  for I := 0 to High(Ids) do
  begin
    Piece := TokenBytes(Ids[I]);
    if Length(Piece) > 0 then
    begin
      SetLength(Buf, N + Length(Piece));
      Move(Piece[0], Buf[N], Length(Piece));
      Inc(N, Length(Piece));
    end;
  end;
  Result := BytesToUtf8Lossy(Buf);
end;

function TLlmTokenizerBase.IsStopToken(Id: Integer): Boolean;
begin
  Result := Id = EosId;
end;

function TLlmTokenizerBase.PrependBos: Boolean;
begin
  Result := False;
end;

{ TLlmBackend }

function TLlmBackend.DefaultTemplate: TChatTemplate;
begin
  Result := ctAuto;
end;

{ Helpers }

function BytesToUtf8Lossy(const B: TBytes): string;
begin
  if Length(B) = 0 then
    Exit('');
  try
    Result := TEncoding.UTF8.GetString(B);
  except
    Result := TEncoding.ANSI.GetString(B);
  end;
end;

{ Length of the longest prefix that contains only COMPLETE UTF-8 sequences.
  Used during token streaming to avoid sending partial characters. }
function Utf8CompletePrefixLength(const B: TBytes): Integer;
var
  N, I, Need: Integer;
  Lead: Byte;
begin
  N := Length(B);
  if N = 0 then
    Exit(0);
  I := N - 1;
  { Scan backwards for the last lead byte (at most 3 continuation bytes) }
  while (I > 0) and (I > N - 4) and ((B[I] and $C0) = $80) do
    Dec(I);
  Lead := B[I];
  if Lead < $80 then
    Need := 1
  else if (Lead and $E0) = $C0 then
    Need := 2
  else if (Lead and $F0) = $E0 then
    Need := 3
  else if (Lead and $F8) = $F0 then
    Need := 4
  else
    Need := 1; // invalid -> treat as complete
  if I + Need <= N then
    Result := N
  else
    Result := I;
end;

end.
