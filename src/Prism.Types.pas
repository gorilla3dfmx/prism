unit Prism.Types;

{ Prism - Ein LLM-Framework in reinem Object Pascal (Delphi 13).
  Basistypen, Konfiguration, Parameter-Layout, RNG und die abstrakten
  Basisklassen fuer Engines/Tokenizer/Backends.

  WICHTIG: Alle Groessen/Offsets sind Int64-faehig (NativeInt auf 64-Bit),
  damit Modelle mit Milliarden Parametern adressierbar sind.
  Fuer grosse Modelle immer 64-Bit-Targets (Win64, Linux64, Android64,
  iOS/macOS ARM64) kompilieren. }

interface

uses
  System.SysUtils;

const
  PRISM_VERSION = '0.1.0';

  { Eigenes Checkpoint-Format (trainierbare Modelle) }
  CHECKPOINT_MAGIC: UInt32 = $4D535250; // 'PRSM' little-endian
  CHECKPOINT_VERSION = 1;
  CHECKPOINT_HEADER_SIZE = 64;

type
  { Konfiguration des eigenen (trainierbaren) GPT-2-artigen Modells.
    NumExperts > 1 aktiviert Mixture-of-Experts ("thematische Areale"):
    ein Router waehlt pro Token EINEN Experten (Top-1) - nur dieser
    Untergraph wird gerechnet und muss im Speicher liegen. }
  TModelConfig = record
    VocabSize: Integer;
    SeqLen: Integer;      // maximale Kontextlaenge
    Dim: Integer;         // Embedding-Dimension C
    NumLayers: Integer;
    NumHeads: Integer;
    NumExperts: Integer;  // 1 = klassisches dichtes FFN
    function Hidden: Integer;   // MLP hidden = 4*C (pro Experte)
    function HeadSize: Integer; // C div NumHeads
    function IsMoE: Boolean;
    function ToString: string;
  end;

  { Flat-Layout aller Parameter. Reihenfolge:
    [wte][wpe][lnf_w][lnf_b][layer 0][layer 1]...[layer L-1]
    Layer-Block: [core: ln1, qkv, proj, ln2, router][expert 0]...[expert E-1]
    Der "Resident"-Prefix (Embeddings + finale Norm) bleibt beim Streaming
    immer im Speicher; Layer-Cores und einzelne Experten werden als
    Cluster bedarfsweise nachgeladen (LRU). }
  TParamLayout = record
    Config: TModelConfig;
    OffWte, OffWpe, OffLnfW, OffLnfB: Int64;
    ResidentCount: Int64;   // Anzahl Floats im Resident-Prefix
    LayerCoreCount: Int64;  // Floats im Layer-Core (Attention + Router)
    ExpertCount: Int64;     // Floats pro Experte (FFN)
    LayerCount: Int64;      // Core + NumExperts * ExpertCount
    TotalCount: Int64;
    { Offsets INNERHALB des Layer-Cores (relativ zur Layer-Basis) }
    RLn1W, RLn1B, RQkvW, RQkvB, RProjW, RProjB,
    RLn2W, RLn2B, RRouterW: Int64;
    { Offsets INNERHALB eines Experten-Blocks (relativ zur Experten-Basis) }
    XFcW, XFcB, XFc2W, XFc2B: Int64;
    procedure Init(const AConfig: TModelConfig);
    function LayerBase(L: Integer): Int64;
    function ExpertBase(L, E: Integer): Int64;
  end;

  { Deterministischer RNG (xorshift64*), unabhaengig vom System-Random }
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

  { Abstrakter Tokenizer: implementiert von TTokenizer (eigenes Byte-BPE),
    TSpmTokenizer und TGpt2Tokenizer (beide aus GGUF-Metadaten). }
  TLlmTokenizerBase = class
  public
    function VocabSize: Integer; virtual; abstract;
    function Encode(const Text: string): TArray<Integer>; virtual; abstract;
    function TokenBytes(Id: Integer): TBytes; virtual; abstract;
    function Decode(const Ids: TArray<Integer>): string; virtual;
    function BosId: Integer; virtual; abstract;
    function EosId: Integer; virtual; abstract;
    function IsStopToken(Id: Integer): Boolean; virtual;
    { True, wenn Roh-Prompts (completion/generate) ein BOS vorangestellt
      werden soll (GGUF-Modelle erwarten das; Prism-Korpusformat nicht) }
    function PrependBos: Boolean; virtual;
    { Baut aus Chat-Nachrichten die Prompt-Token-Sequenz (inkl. Template) }
    function BuildChatTokens(const Msgs: TChatMessages;
      Template: TChatTemplate): TArray<Integer>; virtual; abstract;
  end;

  { Abstrakte Inferenz-Engine (ein Exemplar pro Request; haelt KV-Cache).
    Step verarbeitet EIN Token; NeedLogits=False beim Prompt-Prefill spart
    die teure Logits-Projektion (V x C) - wichtiger Performance-Gewinn. }
  TLlmEngine = class
  public
    procedure Reset; virtual; abstract;
    procedure Step(Token: Integer; NeedLogits: Boolean); virtual; abstract;
    function Logits: TArray<Single>; virtual; abstract;
    function VocabSize: Integer; virtual; abstract;
    function MaxContext: Integer; virtual; abstract;
    function Position: Integer; virtual; abstract;
  end;

  { Backend = geladenes Modell + Tokenizer; Factory fuer Engines }
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
  { Layer-Core }
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
    Inc(O, E * C); // Router nur bei MoE
  LayerCoreCount := O;
  { Experten-Block }
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

{ Laenge des laengsten Prefix, das nur VOLLSTAENDIGE UTF-8-Sequenzen enthaelt.
  Wird beim Token-Streaming genutzt, um keine halben Zeichen zu senden. }
function Utf8CompletePrefixLength(const B: TBytes): Integer;
var
  N, I, Need: Integer;
  Lead: Byte;
begin
  N := Length(B);
  if N = 0 then
    Exit(0);
  I := N - 1;
  { Rueckwaerts das letzte Lead-Byte suchen (max. 3 Continuation-Bytes) }
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
    Need := 1; // ungueltig -> als komplett behandeln
  if I + Need <= N then
    Result := N
  else
    Result := I;
end;

end.
