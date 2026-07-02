unit Prism.Model;

{ Prism's own checkpoint format (.prism) and weights providers.

  File format:
    Header (64 bytes): magic 'PRSM', version, config
    followed by all parameters as Float32 in the TParamLayout flat layout.

  TWeightsProvider abstracts the access:
    - TFullWeights: everything in RAM (training, fastest inference)
    - TLayerStore (Prism.Streaming): layer/expert clusters from disk (LRU) }

interface

uses
  System.SysUtils, System.Classes, Prism.Types;

type
  TCheckpointHeader = packed record
    Magic: UInt32;
    Version: Int32;
    VocabSize: Int32;
    SeqLen: Int32;
    Dim: Int32;
    NumLayers: Int32;
    NumHeads: Int32;
    NumExperts: Int32;
    Pad: array [0 .. 31] of Byte;
  end;

  TWeightsProvider = class
  public
    Config: TModelConfig;
    Layout: TParamLayout;
    { Contains at least the resident prefix (wte, wpe, lnf) at the
      layout offsets. For TFullWeights it is identical to Params. }
    Resident: TArray<Single>;
    { Layer core: Arr[Base + Layout.R*] }
    procedure GetLayer(L: Integer; out Arr: TArray<Single>;
      out Base: Int64); virtual; abstract;
    { Expert: Arr[Base + Layout.X*] }
    procedure GetExpert(L, E: Integer; out Arr: TArray<Single>;
      out Base: Int64); virtual; abstract;
  end;

  TFullWeights = class(TWeightsProvider)
  public
    Params: TArray<Single>;
    procedure GetLayer(L: Integer; out Arr: TArray<Single>;
      out Base: Int64); override;
    procedure GetExpert(L, E: Integer; out Arr: TArray<Single>;
      out Base: Int64); override;
    procedure InitRandom(const AConfig: TModelConfig; Seed: UInt64);
    procedure SaveToFile(const Path: string);
    procedure LoadFromFile(const Path: string);
  end;

function ReadCheckpointConfig(const Path: string;
  out AConfig: TModelConfig): Boolean;
procedure WriteCheckpointHeader(Stream: TStream; const AConfig: TModelConfig);
function ReadCheckpointHeader(Stream: TStream): TModelConfig;

implementation

uses
  System.Math;

procedure WriteCheckpointHeader(Stream: TStream; const AConfig: TModelConfig);
var
  H: TCheckpointHeader;
begin
  FillChar(H, SizeOf(H), 0);
  H.Magic := CHECKPOINT_MAGIC;
  H.Version := CHECKPOINT_VERSION;
  H.VocabSize := AConfig.VocabSize;
  H.SeqLen := AConfig.SeqLen;
  H.Dim := AConfig.Dim;
  H.NumLayers := AConfig.NumLayers;
  H.NumHeads := AConfig.NumHeads;
  H.NumExperts := AConfig.NumExperts;
  Assert(SizeOf(H) = CHECKPOINT_HEADER_SIZE);
  Stream.WriteBuffer(H, SizeOf(H));
end;

function ReadCheckpointHeader(Stream: TStream): TModelConfig;
var
  H: TCheckpointHeader;
begin
  Stream.ReadBuffer(H, SizeOf(H));
  if H.Magic <> CHECKPOINT_MAGIC then
    raise Exception.Create('Not a valid Prism checkpoint file (magic).');
  if H.Version <> CHECKPOINT_VERSION then
    raise Exception.CreateFmt('Checkpoint version %d not supported.',
      [H.Version]);
  Result.VocabSize := H.VocabSize;
  Result.SeqLen := H.SeqLen;
  Result.Dim := H.Dim;
  Result.NumLayers := H.NumLayers;
  Result.NumHeads := H.NumHeads;
  Result.NumExperts := H.NumExperts;
  if Result.NumExperts < 1 then
    Result.NumExperts := 1;
end;

function ReadCheckpointConfig(const Path: string;
  out AConfig: TModelConfig): Boolean;
var
  FS: TFileStream;
begin
  Result := False;
  if not FileExists(Path) then
    Exit;
  FS := TFileStream.Create(Path, fmOpenRead or fmShareDenyWrite);
  try
    try
      AConfig := ReadCheckpointHeader(FS);
      Result := True;
    except
      Result := False;
    end;
  finally
    FS.Free;
  end;
end;

{ TFullWeights }

procedure TFullWeights.GetLayer(L: Integer; out Arr: TArray<Single>;
  out Base: Int64);
begin
  Arr := Params;
  Base := Layout.LayerBase(L);
end;

procedure TFullWeights.GetExpert(L, E: Integer; out Arr: TArray<Single>;
  out Base: Int64);
begin
  Arr := Params;
  Base := Layout.ExpertBase(L, E);
end;

procedure TFullWeights.InitRandom(const AConfig: TModelConfig; Seed: UInt64);
var
  Rng: TRng;
  L, E: Integer;
  I, LB, XB: Int64;
  C, H: Integer;
  Std, ResidStd: Single;

  procedure FillGauss(Off, Count: Int64; S: Single);
  var
    J: Int64;
  begin
    for J := Off to Off + Count - 1 do
      Params[J] := Rng.NextGauss * S;
  end;

  procedure FillConst(Off, Count: Int64; V: Single);
  var
    J: Int64;
  begin
    for J := Off to Off + Count - 1 do
      Params[J] := V;
  end;

begin
  Config := AConfig;
  if Config.NumExperts < 1 then
    Config.NumExperts := 1;
  Layout.Init(Config);
  SetLength(Params, Layout.TotalCount);
  Resident := Params;
  Rng.Seed(Seed);
  C := Config.Dim;
  H := Config.Hidden;
  Std := 0.02;
  { GPT-2 trick: initialize residual projections with a smaller scale }
  ResidStd := 0.02 / Sqrt(2.0 * Config.NumLayers);

  FillGauss(Layout.OffWte, Int64(Config.VocabSize) * C, Std);
  FillGauss(Layout.OffWpe, Int64(Config.SeqLen) * C, Std);
  FillConst(Layout.OffLnfW, C, 1.0);
  FillConst(Layout.OffLnfB, C, 0.0);

  for L := 0 to Config.NumLayers - 1 do
  begin
    LB := Layout.LayerBase(L);
    FillConst(LB + Layout.RLn1W, C, 1.0);
    FillConst(LB + Layout.RLn1B, C, 0.0);
    FillGauss(LB + Layout.RQkvW, 3 * Int64(C) * C, Std);
    FillConst(LB + Layout.RQkvB, 3 * C, 0.0);
    FillGauss(LB + Layout.RProjW, Int64(C) * C, ResidStd);
    FillConst(LB + Layout.RProjB, C, 0.0);
    FillConst(LB + Layout.RLn2W, C, 1.0);
    FillConst(LB + Layout.RLn2B, C, 0.0);
    if Config.IsMoE then
      FillGauss(LB + Layout.RRouterW, Int64(Config.NumExperts) * C, Std);
    for E := 0 to Config.NumExperts - 1 do
    begin
      XB := Layout.ExpertBase(L, E);
      FillGauss(XB + Layout.XFcW, Int64(H) * C, Std);
      FillConst(XB + Layout.XFcB, H, 0.0);
      FillGauss(XB + Layout.XFc2W, Int64(C) * H, ResidStd);
      FillConst(XB + Layout.XFc2B, C, 0.0);
    end;
  end;
  { Starting Wpe somewhat smaller helps stability for small models }
  for I := Layout.OffWpe to Layout.OffWpe + Int64(Config.SeqLen) * C - 1 do
    Params[I] := Params[I] * 0.5;
end;

procedure TFullWeights.SaveToFile(const Path: string);
var
  FS: TFileStream;
  Written, ChunkFloats: Int64;
const
  MAX_CHUNK = 64 * 1024 * 1024; // write in 64M-float blocks
begin
  FS := TFileStream.Create(Path, fmCreate);
  try
    WriteCheckpointHeader(FS, Config);
    Written := 0;
    while Written < Layout.TotalCount do
    begin
      ChunkFloats := Min(Int64(MAX_CHUNK), Layout.TotalCount - Written);
      FS.WriteBuffer(Params[Written], ChunkFloats * SizeOf(Single));
      Inc(Written, ChunkFloats);
    end;
  finally
    FS.Free;
  end;
end;

procedure TFullWeights.LoadFromFile(const Path: string);
var
  FS: TFileStream;
  ReadN, ChunkFloats: Int64;
const
  MAX_CHUNK = 64 * 1024 * 1024;
begin
  FS := TFileStream.Create(Path, fmOpenRead or fmShareDenyWrite);
  try
    Config := ReadCheckpointHeader(FS);
    Layout.Init(Config);
    SetLength(Params, Layout.TotalCount);
    Resident := Params;
    ReadN := 0;
    while ReadN < Layout.TotalCount do
    begin
      ChunkFloats := Min(Int64(MAX_CHUNK), Layout.TotalCount - ReadN);
      FS.ReadBuffer(Params[ReadN], ChunkFloats * SizeOf(Single));
      Inc(ReadN, ChunkFloats);
    end;
  finally
    FS.Free;
  end;
end;

end.
