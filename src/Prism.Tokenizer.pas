unit Prism.Tokenizer;

{ Custom byte-level BPE tokenizer for trainable Prism models.

  Byte-level = ANY kind of data is tokenizable (text, images, audio, video,
  3D data, binary data) - this is the foundation of multimodality:
  the base vocabulary is the 256 bytes, on top of that learned BPE merges,
  followed by fixed special tokens (chat roles and modality markers). }

interface

uses
  System.SysUtils, System.Classes, System.IOUtils, System.Generics.Collections,
  System.JSON, Prism.Types;

const
  SPECIAL_TOKENS: array [0 .. 15] of string = (
    '<|pad|>', '<|bos|>', '<|eos|>', '<|system|>', '<|user|>', '<|assistant|>',
    '<|img|>', '<|/img|>', '<|aud|>', '<|/aud|>', '<|vid|>', '<|/vid|>',
    '<|3d|>', '<|/3d|>', '<|bin|>', '<|/bin|>');

  SP_PAD = 0; SP_BOS = 1; SP_EOS = 2; SP_SYSTEM = 3; SP_USER = 4;
  SP_ASSISTANT = 5;

type
  TTokenizer = class(TLlmTokenizerBase)
  private
    FMergeA, FMergeB: TArray<Integer>;      // Merge i: (A,B) -> 256+i
    FPairToId: TDictionary<Int64, Integer>;
    FDecodeTab: TArray<TBytes>;             // Id < SpecialBase -> Bytes
    FSpecialBytes: TArray<TBytes>;
    procedure RebuildTables;
    function FindSpecialAt(const Data: TBytes; Pos: Integer): Integer;
    procedure ApplyBpe(const Data: TBytes; From, Count: Integer;
      Res: TList<Integer>);
    class function PairKey(A, B: Integer): Int64; static; inline;
    class procedure MergePass(var Ids: TArray<Integer>; var N: Integer;
      A, B, NewId: Integer); static;
  public
    constructor Create;
    destructor Destroy; override;

    function SpecialBase: Integer;
    function VocabSize: Integer; override;
    function SpecialId(Index: Integer): Integer; // Index in SPECIAL_TOKENS
    function IsSpecial(Id: Integer): Boolean;
    function BosId: Integer; override;
    function EosId: Integer; override;
    function IsStopToken(Id: Integer): Boolean; override;

    { Learn BPE merges from raw data. TargetVocab = 256 + merges + specials }
    procedure TrainFromData(const Data: TBytes; TargetVocab: Integer;
      MaxBytes: Integer; const Log: TProc<string>);

    function EncodeData(const Data: TBytes): TArray<Integer>;
    function Encode(const Text: string): TArray<Integer>; override;
    function TokenBytes(Id: Integer): TBytes; override;
    function BuildChatTokens(const Msgs: TChatMessages;
      Template: TChatTemplate): TArray<Integer>; override;

    procedure SaveToFile(const Path: string);
    procedure LoadFromFile(const Path: string);
  end;

implementation

constructor TTokenizer.Create;
begin
  inherited Create;
  FPairToId := TDictionary<Int64, Integer>.Create;
  RebuildTables;
end;

destructor TTokenizer.Destroy;
begin
  FPairToId.Free;
  inherited;
end;

class function TTokenizer.PairKey(A, B: Integer): Int64;
begin
  Result := (Int64(A) shl 32) or Int64(UInt32(B));
end;

function TTokenizer.SpecialBase: Integer;
begin
  Result := 256 + Length(FMergeA);
end;

function TTokenizer.VocabSize: Integer;
begin
  Result := SpecialBase + Length(SPECIAL_TOKENS);
end;

function TTokenizer.SpecialId(Index: Integer): Integer;
begin
  Result := SpecialBase + Index;
end;

function TTokenizer.IsSpecial(Id: Integer): Boolean;
begin
  Result := Id >= SpecialBase;
end;

function TTokenizer.BosId: Integer;
begin
  Result := SpecialId(SP_BOS);
end;

function TTokenizer.EosId: Integer;
begin
  Result := SpecialId(SP_EOS);
end;

function TTokenizer.IsStopToken(Id: Integer): Boolean;
begin
  { Every special token stops generation (role switch etc.) }
  Result := IsSpecial(Id) and (Id <> SpecialId(SP_PAD));
end;

procedure TTokenizer.RebuildTables;
var
  I: Integer;
  A, B: TBytes;
begin
  FPairToId.Clear;
  SetLength(FDecodeTab, SpecialBase);
  for I := 0 to 255 do
  begin
    SetLength(FDecodeTab[I], 1);
    FDecodeTab[I][0] := Byte(I);
  end;
  for I := 0 to High(FMergeA) do
  begin
    FPairToId.AddOrSetValue(PairKey(FMergeA[I], FMergeB[I]), 256 + I);
    A := FDecodeTab[FMergeA[I]];
    B := FDecodeTab[FMergeB[I]];
    SetLength(FDecodeTab[256 + I], Length(A) + Length(B));
    if Length(A) > 0 then
      Move(A[0], FDecodeTab[256 + I][0], Length(A));
    if Length(B) > 0 then
      Move(B[0], FDecodeTab[256 + I][Length(A)], Length(B));
  end;
  SetLength(FSpecialBytes, Length(SPECIAL_TOKENS));
  for I := 0 to High(SPECIAL_TOKENS) do
    FSpecialBytes[I] := TEncoding.UTF8.GetBytes(SPECIAL_TOKENS[I]);
end;

function TTokenizer.TokenBytes(Id: Integer): TBytes;
begin
  if (Id >= 0) and (Id < SpecialBase) then
    Result := FDecodeTab[Id]
  else if (Id >= SpecialBase) and (Id < VocabSize) then
    Result := FSpecialBytes[Id - SpecialBase]
  else
    Result := nil;
end;

function TTokenizer.FindSpecialAt(const Data: TBytes; Pos: Integer): Integer;
var
  K, I, L: Integer;
  Match: Boolean;
begin
  Result := -1;
  for K := 0 to High(FSpecialBytes) do
  begin
    L := Length(FSpecialBytes[K]);
    if Pos + L <= Length(Data) then
    begin
      Match := True;
      for I := 0 to L - 1 do
        if Data[Pos + I] <> FSpecialBytes[K][I] then
        begin
          Match := False;
          Break;
        end;
      if Match then
        Exit(K);
    end;
  end;
end;

class procedure TTokenizer.MergePass(var Ids: TArray<Integer>; var N: Integer;
  A, B, NewId: Integer);
var
  R, W: Integer;
begin
  R := 0;
  W := 0;
  while R < N do
  begin
    if (R < N - 1) and (Ids[R] = A) and (Ids[R + 1] = B) then
    begin
      Ids[W] := NewId;
      Inc(R, 2);
    end
    else
    begin
      Ids[W] := Ids[R];
      Inc(R);
    end;
    Inc(W);
  end;
  N := W;
end;

procedure TTokenizer.ApplyBpe(const Data: TBytes; From, Count: Integer;
  Res: TList<Integer>);
var
  Ids: TArray<Integer>;
  N, I, Mid, BestId, BestA, BestB: Integer;
begin
  if Count <= 0 then
    Exit;
  SetLength(Ids, Count);
  for I := 0 to Count - 1 do
    Ids[I] := Data[From + I];
  N := Count;
  while N > 1 do
  begin
    BestId := MaxInt;
    BestA := -1;
    BestB := -1;
    for I := 0 to N - 2 do
      if FPairToId.TryGetValue(PairKey(Ids[I], Ids[I + 1]), Mid) and
        (Mid < BestId) then
      begin
        BestId := Mid;
        BestA := Ids[I];
        BestB := Ids[I + 1];
      end;
    if BestA < 0 then
      Break;
    MergePass(Ids, N, BestA, BestB, BestId);
  end;
  for I := 0 to N - 1 do
    Res.Add(Ids[I]);
end;

function TTokenizer.EncodeData(const Data: TBytes): TArray<Integer>;
var
  Res: TList<Integer>;
  I, SegStart, K: Integer;
begin
  Res := TList<Integer>.Create;
  try
    SegStart := 0;
    I := 0;
    while I < Length(Data) do
    begin
      if Data[I] = Ord('<') then
      begin
        K := FindSpecialAt(Data, I);
        if K >= 0 then
        begin
          ApplyBpe(Data, SegStart, I - SegStart, Res);
          Res.Add(SpecialBase + K);
          Inc(I, Length(FSpecialBytes[K]));
          SegStart := I;
          Continue;
        end;
      end;
      Inc(I);
    end;
    ApplyBpe(Data, SegStart, Length(Data) - SegStart, Res);
    Result := Res.ToArray;
  finally
    Res.Free;
  end;
end;

function TTokenizer.Encode(const Text: string): TArray<Integer>;
begin
  Result := EncodeData(TEncoding.UTF8.GetBytes(Text));
end;

function TTokenizer.BuildChatTokens(const Msgs: TChatMessages;
  Template: TChatTemplate): TArray<Integer>;
var
  Res: TList<Integer>;
  M: TChatMessage;
  RoleId: Integer;
begin
  Res := TList<Integer>.Create;
  try
    { No BOS: the Prism corpus format starts directly with <|user|> }
    for M in Msgs do
    begin
      if SameText(M.Role, 'system') then
        RoleId := SpecialId(SP_SYSTEM)
      else if SameText(M.Role, 'assistant') then
        RoleId := SpecialId(SP_ASSISTANT)
      else
        RoleId := SpecialId(SP_USER);
      Res.Add(RoleId);
      Res.AddRange(Encode(M.Content));
      if RoleId = SpecialId(SP_ASSISTANT) then
        Res.Add(EosId);
    end;
    Res.Add(SpecialId(SP_ASSISTANT));
    Result := Res.ToArray;
  finally
    Res.Free;
  end;
end;

procedure TTokenizer.TrainFromData(const Data: TBytes; TargetVocab: Integer;
  MaxBytes: Integer; const Log: TProc<string>);
var
  Ids: TArray<Integer>;
  N, NumMerges, M, I, K, SegStart, Count, BestCount, BestA, BestB: Integer;
  Pairs: TDictionary<Int64, Integer>;
  Key: Int64;
  Pair: TPair<Int64, Integer>;
  Src: TBytes;
begin
  Src := Data;
  if (MaxBytes > 0) and (Length(Src) > MaxBytes) then
    SetLength(Src, MaxBytes);
  NumMerges := TargetVocab - 256 - Length(SPECIAL_TOKENS);
  if NumMerges < 0 then
    NumMerges := 0;
  SetLength(FMergeA, 0);
  SetLength(FMergeB, 0);
  RebuildTables;

  { Build the id stream, -1 as separator at special-token boundaries
    (merging must not cross markers) }
  SetLength(Ids, Length(Src));
  N := 0;
  I := 0;
  SegStart := 0;
  while I < Length(Src) do
  begin
    if Src[I] = Ord('<') then
    begin
      K := FindSpecialAt(Src, I);
      if K >= 0 then
      begin
        while SegStart < I do
        begin
          Ids[N] := Src[SegStart];
          Inc(N);
          Inc(SegStart);
        end;
        Ids[N] := -1;
        Inc(N);
        Inc(I, Length(FSpecialBytes[K]));
        SegStart := I;
        Continue;
      end;
    end;
    Inc(I);
  end;
  while SegStart < Length(Src) do
  begin
    Ids[N] := Src[SegStart];
    Inc(N);
    Inc(SegStart);
  end;

  Pairs := TDictionary<Int64, Integer>.Create;
  try
    for M := 0 to NumMerges - 1 do
    begin
      Pairs.Clear;
      for I := 0 to N - 2 do
        if (Ids[I] >= 0) and (Ids[I + 1] >= 0) then
        begin
          Key := PairKey(Ids[I], Ids[I + 1]);
          if Pairs.TryGetValue(Key, Count) then
            Pairs[Key] := Count + 1
          else
            Pairs.Add(Key, 1);
        end;
      BestCount := 1;
      BestA := -1;
      BestB := -1;
      for Pair in Pairs do
        if Pair.Value > BestCount then
        begin
          BestCount := Pair.Value;
          BestA := Integer(Pair.Key shr 32);
          BestB := Integer(UInt32(Pair.Key and $FFFFFFFF));
        end;
      if BestA < 0 then
        Break; // no more repeated pairs
      SetLength(FMergeA, Length(FMergeA) + 1);
      SetLength(FMergeB, Length(FMergeB) + 1);
      FMergeA[High(FMergeA)] := BestA;
      FMergeB[High(FMergeB)] := BestB;
      MergePass(Ids, N, BestA, BestB, 256 + M);
      if Assigned(Log) and ((M mod 25 = 0) or (M = NumMerges - 1)) then
        Log(Format('Merge %d/%d: (%d,%d) x%d, stream=%d tokens',
          [M + 1, NumMerges, BestA, BestB, BestCount, N]));
    end;
  finally
    Pairs.Free;
  end;
  RebuildTables;
end;

procedure TTokenizer.SaveToFile(const Path: string);
var
  Root: TJSONObject;
  Arr, P: TJSONArray;
  I: Integer;
begin
  Root := TJSONObject.Create;
  try
    Root.AddPair('version', TJSONNumber.Create(1));
    Root.AddPair('type', 'prism-byte-bpe');
    Arr := TJSONArray.Create;
    for I := 0 to High(FMergeA) do
    begin
      P := TJSONArray.Create;
      P.Add(FMergeA[I]);
      P.Add(FMergeB[I]);
      Arr.AddElement(P);
    end;
    Root.AddPair('merges', Arr);
    TFile.WriteAllText(Path, Root.ToJSON, TEncoding.UTF8);
  finally
    Root.Free;
  end;
end;

procedure TTokenizer.LoadFromFile(const Path: string);
var
  Root: TJSONValue;
  Arr: TJSONArray;
  I: Integer;
  P: TJSONArray;
begin
  Root := TJSONObject.ParseJSONValue(TFile.ReadAllText(Path, TEncoding.UTF8));
  if Root = nil then
    raise Exception.Create('Tokenizer: could not parse JSON: ' + Path);
  try
    Arr := Root.GetValue<TJSONArray>('merges');
    SetLength(FMergeA, Arr.Count);
    SetLength(FMergeB, Arr.Count);
    for I := 0 to Arr.Count - 1 do
    begin
      P := Arr.Items[I] as TJSONArray;
      FMergeA[I] := P.Items[0].GetValue<Integer>;
      FMergeB[I] := P.Items[1].GetValue<Integer>;
    end;
  finally
    Root.Free;
  end;
  RebuildTables;
end;

end.
