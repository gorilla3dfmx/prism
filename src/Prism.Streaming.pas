unit Prism.Streaming;

{ Cluster-Streaming fuer .prism-Checkpoints: statt das komplette Modell in
  den Speicher zu laden, werden nur der Resident-Prefix (Embeddings + finale
  Norm) dauerhaft gehalten. Layer-Cores und einzelne FFN-Experten
  ("thematische Areale") werden bedarfsweise von Platte gelesen und in
  LRU-Caches gehalten.

  Speicherbedarf ~ Resident + MaxLayers * LayerCore + MaxExperts * Expert
  statt der vollen Parameterzahl - entscheidend fuer mobile Geraete.
  Bei MoE wird pro Token ohnehin nur EIN Experte pro Layer angefasst,
  d.h. haeufig genutzte Areale bleiben im Cache "warm". }

interface

uses
  System.SysUtils, System.Classes, System.SyncObjs,
  System.Generics.Collections, Prism.Types, Prism.Model, Prism.Gpu;

type
  TLayerStore = class(TWeightsProvider)
  private
    FStream: TFileStream;
    FLock: TCriticalSection;
    FCoreCache: TDictionary<Integer, TArray<Single>>;
    FCoreOrder: TList<Integer>;
    FExpertCache: TDictionary<Int64, TArray<Single>>;
    FExpertOrder: TList<Int64>;
    FMaxLayers: Integer;
    FMaxExperts: Integer;
    FCoreReads, FCoreHits, FExpertReads, FExpertHits: Int64;
    function ReadFloats(FileOffFloats: Int64; Count: Int64): TArray<Single>;
    class function ExpertKey(L, E: Integer): Int64; static; inline;
  public
    { MaxCachedLayers/MaxCachedExperts steuern den Speicherverbrauch.
      MaxCachedExperts gilt ueber alle Layer hinweg. }
    constructor Create(const Path: string; MaxCachedLayers: Integer = 4;
      MaxCachedExperts: Integer = 8);
    destructor Destroy; override;
    procedure GetLayer(L: Integer; out Arr: TArray<Single>;
      out Base: Int64); override;
    procedure GetExpert(L, E: Integer; out Arr: TArray<Single>;
      out Base: Int64); override;
    function StatsText: string;
  end;

implementation

constructor TLayerStore.Create(const Path: string; MaxCachedLayers: Integer;
  MaxCachedExperts: Integer);
begin
  inherited Create;
  FLock := TCriticalSection.Create;
  FCoreCache := TDictionary<Integer, TArray<Single>>.Create;
  FCoreOrder := TList<Integer>.Create;
  FExpertCache := TDictionary<Int64, TArray<Single>>.Create;
  FExpertOrder := TList<Int64>.Create;
  FMaxLayers := MaxCachedLayers;
  if FMaxLayers < 1 then
    FMaxLayers := 1;
  FMaxExperts := MaxCachedExperts;
  if FMaxExperts < 1 then
    FMaxExperts := 1;
  FStream := TFileStream.Create(Path, fmOpenRead or fmShareDenyWrite);
  Config := ReadCheckpointHeader(FStream);
  Layout.Init(Config);
  Resident := ReadFloats(0, Layout.ResidentCount);
end;

destructor TLayerStore.Destroy;
begin
  FStream.Free;
  FCoreCache.Free;
  FCoreOrder.Free;
  FExpertCache.Free;
  FExpertOrder.Free;
  FLock.Free;
  inherited;
end;

class function TLayerStore.ExpertKey(L, E: Integer): Int64;
begin
  Result := (Int64(L) shl 32) or Int64(UInt32(E));
end;

function TLayerStore.ReadFloats(FileOffFloats: Int64; Count: Int64):
  TArray<Single>;
begin
  SetLength(Result, Count);
  FStream.Position := CHECKPOINT_HEADER_SIZE + FileOffFloats * SizeOf(Single);
  FStream.ReadBuffer(Result[0], Count * SizeOf(Single));
end;

procedure TLayerStore.GetLayer(L: Integer; out Arr: TArray<Single>;
  out Base: Int64);
var
  Doomed: TArray<Single>;
begin
  Base := 0;
  FLock.Enter;
  try
    Inc(FCoreReads);
    if FCoreCache.TryGetValue(L, Arr) then
    begin
      Inc(FCoreHits);
      FCoreOrder.Remove(L);
      FCoreOrder.Add(L);
      Exit;
    end;
    Arr := ReadFloats(Layout.LayerBase(L), Layout.LayerCoreCount);
    FCoreCache.Add(L, Arr);
    FCoreOrder.Add(L);
    while FCoreOrder.Count > FMaxLayers do
    begin
      if FCoreCache.TryGetValue(FCoreOrder[0], Doomed) then
        Backend.InvalidateWeights(Pointer(Doomed)); // GPU-Buffer verwerfen
      FCoreCache.Remove(FCoreOrder[0]);
      FCoreOrder.Delete(0);
    end;
  finally
    FLock.Leave;
  end;
end;

procedure TLayerStore.GetExpert(L, E: Integer; out Arr: TArray<Single>;
  out Base: Int64);
var
  Key: Int64;
  Doomed: TArray<Single>;
begin
  Base := 0;
  Key := ExpertKey(L, E);
  FLock.Enter;
  try
    Inc(FExpertReads);
    if FExpertCache.TryGetValue(Key, Arr) then
    begin
      Inc(FExpertHits);
      FExpertOrder.Remove(Key);
      FExpertOrder.Add(Key);
      Exit;
    end;
    Arr := ReadFloats(Layout.ExpertBase(L, E), Layout.ExpertCount);
    FExpertCache.Add(Key, Arr);
    FExpertOrder.Add(Key);
    while FExpertOrder.Count > FMaxExperts do
    begin
      if FExpertCache.TryGetValue(FExpertOrder[0], Doomed) then
        Backend.InvalidateWeights(Pointer(Doomed));
      FExpertCache.Remove(FExpertOrder[0]);
      FExpertOrder.Delete(0);
    end;
  finally
    FLock.Leave;
  end;
end;

function TLayerStore.StatsText: string;

  function Pct(Hits, Reads: Int64): string;
  begin
    if Reads = 0 then
      Result := '-'
    else
      Result := Format('%.1f%%', [100.0 * Hits / Reads]);
  end;

begin
  FLock.Enter;
  try
    Result := Format('LayerCache: %d/%d (Hitrate %s), ExpertCache: %d/%d (Hitrate %s)',
      [FCoreCache.Count, FMaxLayers, Pct(FCoreHits, FCoreReads),
       FExpertCache.Count, FMaxExperts, Pct(FExpertHits, FExpertReads)]);
  finally
    FLock.Leave;
  end;
end;

end.
