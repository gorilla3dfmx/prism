unit Prism.Gguf;

{ GGUF-Reader in reinem Object Pascal: laedt existierende, bereits
  trainierte LLMs aus dem llama.cpp-Oekosystem (TinyLlama, Llama, Mistral,
  Qwen, ...) OHNE Konvertierung direkt aus der .gguf-Datei.

  Unterstuetzt:
  - GGUF Version 2 und 3
  - Tensor-Typen F32, F16, Q4_0, Q4_1, Q8_0 (andere -> requantisieren,
    z.B. mit llama-quantize nach Q8_0/Q4_0)
  - eingebettete Tokenizer: 'llama' (SentencePiece) und 'gpt2' (Byte-BPE)

  Alle Offsets/Groessen sind Int64 - Modelle mit Milliarden Parametern
  und Dateien > 4 GB sind vollstaendig adressierbar. Der Stream bleibt
  geoeffnet, damit Prism.Llama Tensoren lagenweise streamen kann. }

interface

uses
  System.SysUtils, System.Classes, System.SyncObjs, System.Character,
  System.Generics.Collections, Prism.Types, Prism.Vector;

type
  TGgufValueKind = (gvInt, gvFloat, gvBool, gvStr, gvStrArr, gvIntArr, gvFltArr);

  TGgufValue = record
    Kind: TGgufValueKind;
    I: Int64;
    F: Double;
    B: Boolean;
    S: string;
    SA: TArray<string>;
    IA: TArray<Int64>;
    FA: TArray<Double>;
  end;

  TGgufTensorInfo = record
    Name: string;
    Typ: TGgmlType;
    Dims: TArray<Int64>; // Dims[0] = Spalten (schnellste Dimension)
    Offset: Int64;       // relativ zum Datenbereich
    function Rows: Int64;
    function Cols: Int64;
  end;

  TGgufFile = class
  private
    FStream: TFileStream;
    FLock: TCriticalSection;
    FMeta: TDictionary<string, TGgufValue>;
    FTensors: TDictionary<string, TGgufTensorInfo>;
    FDataOffset: Int64;
    FArch: string;
    FPath: string;
    function ReadU32: UInt32;
    function ReadU64: UInt64;
    function ReadStr: string;
    function ReadValue(Typ: UInt32): TGgufValue;
  public
    constructor Create(const Path: string);
    destructor Destroy; override;
    property Arch: string read FArch;
    property Path: string read FPath;
    function HasTensor(const Name: string): Boolean;
    function TensorInfo(const Name: string): TGgufTensorInfo;
    { Tensor (roh/quantisiert) laden - threadsicher }
    function LoadTensor(const Name: string): TQTensor;
    { Tensor nach F32 dequantisieren (fuer Norm-Gewichte, Biases) }
    function LoadTensorF32(const Name: string): TArray<Single>;
    function MetaInt(const Key: string; Def: Int64): Int64;
    function MetaFloat(const Key: string; Def: Double): Double;
    function MetaBool(const Key: string; Def: Boolean): Boolean;
    function MetaStr(const Key, Def: string): string;
    function MetaStrArr(const Key: string): TArray<string>;
    function MetaIntArr(const Key: string): TArray<Int64>;
    function MetaFltArr(const Key: string): TArray<Double>;
    function HasKey(const Key: string): Boolean;
  end;

  { Basis fuer GGUF-Tokenizer: gemeinsame Chat-Template-Logik }
  TGgufTokenizerBase = class(TLlmTokenizerBase)
  protected
    FPieces: TArray<string>;
    FTypes: TArray<Integer>;   // llama.cpp: 1=normal 2=unknown 3=control 6=byte
    FVocab: TDictionary<string, Integer>;
    FBos, FEos: Integer;
    FAddBos: Boolean;
    FStopIds: TList<Integer>;
    FAutoTemplate: TChatTemplate;
    function PieceId(const Piece: string): Integer;
    procedure DetectTemplate;
    procedure AppendChat(Res: TList<Integer>; const Msgs: TChatMessages;
      Template: TChatTemplate);
  public
    constructor Create(Gg: TGgufFile);
    destructor Destroy; override;
    function VocabSize: Integer; override;
    function BosId: Integer; override;
    function EosId: Integer; override;
    function IsStopToken(Id: Integer): Boolean; override;
    function PrependBos: Boolean; override;
    function BuildChatTokens(const Msgs: TChatMessages;
      Template: TChatTemplate): TArray<Integer>; override;
    property AutoTemplate: TChatTemplate read FAutoTemplate;
  end;

  { SentencePiece-artiger Tokenizer (tokenizer.ggml.model = 'llama') }
  TSpmTokenizer = class(TGgufTokenizerBase)
  private
    FScores: TArray<Single>;
    FByteTok: array [0 .. 255] of Integer;
    FAddSpacePrefix: Boolean;
  public
    constructor Create(Gg: TGgufFile);
    function Encode(const Text: string): TArray<Integer>; override;
    function TokenBytes(Id: Integer): TBytes; override;
  end;

  { GPT-2-Byte-BPE (tokenizer.ggml.model = 'gpt2', z.B. Qwen) }
  TGpt2Tokenizer = class(TGgufTokenizerBase)
  private
    FMergeRank: TDictionary<string, Integer>;
    FByteToChar: array [0 .. 255] of Char;
    FCharToByte: TDictionary<Char, Byte>;
    procedure EncodeChunk(const Chunk: string; Res: TList<Integer>);
  public
    constructor Create(Gg: TGgufFile);
    destructor Destroy; override;
    function Encode(const Text: string): TArray<Integer>; override;
    function TokenBytes(Id: Integer): TBytes; override;
  end;

function CreateGgufTokenizer(Gg: TGgufFile): TGgufTokenizerBase;

implementation

const
  GGUF_MAGIC = $46554747; // 'GGUF'
  TT_CONTROL = 3;
  TT_BYTE = 6;

function AlignUp(V, A: Int64): Int64;
begin
  if A <= 1 then
    Exit(V);
  Result := ((V + A - 1) div A) * A;
end;

{ TGgufTensorInfo }

function TGgufTensorInfo.Cols: Int64;
begin
  if Length(Dims) > 0 then
    Result := Dims[0]
  else
    Result := 1;
end;

function TGgufTensorInfo.Rows: Int64;
var
  I: Integer;
begin
  Result := 1;
  for I := 1 to High(Dims) do
    Result := Result * Dims[I];
end;

{ TGgufFile }

constructor TGgufFile.Create(const Path: string);
var
  Version: UInt32;
  TensorCount, KvCount, N: UInt64;
  Key: string;
  VTyp: UInt32;
  Info: TGgufTensorInfo;
  NDims: UInt32;
  D: Integer;
  TypRaw: UInt32;
  Alignment: Int64;
begin
  inherited Create;
  FPath := Path;
  FLock := TCriticalSection.Create;
  FMeta := TDictionary<string, TGgufValue>.Create;
  FTensors := TDictionary<string, TGgufTensorInfo>.Create;
  FStream := TFileStream.Create(Path, fmOpenRead or fmShareDenyWrite);
  if ReadU32 <> GGUF_MAGIC then
    raise Exception.Create('Keine GGUF-Datei: ' + Path);
  Version := ReadU32;
  if (Version < 2) or (Version > 3) then
    raise Exception.CreateFmt('GGUF-Version %d nicht unterstuetzt (nur 2/3).',
      [Version]);
  TensorCount := ReadU64;
  KvCount := ReadU64;
  for N := 1 to KvCount do
  begin
    Key := ReadStr;
    VTyp := ReadU32;
    FMeta.AddOrSetValue(Key, ReadValue(VTyp));
  end;
  for N := 1 to TensorCount do
  begin
    Info.Name := ReadStr;
    NDims := ReadU32;
    SetLength(Info.Dims, NDims);
    for D := 0 to Integer(NDims) - 1 do
      Info.Dims[D] := Int64(ReadU64);
    TypRaw := ReadU32;
    case TypRaw of
      0: Info.Typ := gtF32;
      1: Info.Typ := gtF16;
      2: Info.Typ := gtQ4_0;
      3: Info.Typ := gtQ4_1;
      8: Info.Typ := gtQ8_0;
    else
      Info.Typ := TGgmlType(-1); // beim Laden abgelehnt, Metadaten ok
    end;
    Info.Offset := Int64(ReadU64);
    { Nicht unterstuetzte Typen bleiben registriert (Typ = -1);
      die hilfreiche Fehlermeldung kommt erst beim Laden des Tensors. }
    FTensors.AddOrSetValue(Info.Name, Info);
  end;
  Alignment := MetaInt('general.alignment', 32);
  FDataOffset := AlignUp(FStream.Position, Alignment);
  FArch := MetaStr('general.architecture', 'unbekannt');
end;

destructor TGgufFile.Destroy;
begin
  FStream.Free;
  FMeta.Free;
  FTensors.Free;
  FLock.Free;
  inherited;
end;

function TGgufFile.ReadU32: UInt32;
begin
  FStream.ReadBuffer(Result, 4);
end;

function TGgufFile.ReadU64: UInt64;
begin
  FStream.ReadBuffer(Result, 8);
end;

function TGgufFile.ReadStr: string;
var
  Len: UInt64;
  B: TBytes;
begin
  Len := ReadU64;
  if Len = 0 then
    Exit('');
  if Len > 128 * 1024 * 1024 then
    raise Exception.Create('GGUF: String zu lang (Datei defekt?)');
  SetLength(B, Len);
  FStream.ReadBuffer(B[0], Len);
  Result := TEncoding.UTF8.GetString(B);
end;

function TGgufFile.ReadValue(Typ: UInt32): TGgufValue;
var
  U8: Byte;
  I8: ShortInt;
  U16: Word;
  I16: SmallInt;
  U32: UInt32;
  I32: Int32;
  F32: Single;
  F64: Double;
  U64: UInt64;
  I64: Int64;
  B8: Byte;
  ElemTyp: UInt32;
  Count, N: UInt64;
  Elem: TGgufValue;
  SL: TList<string>;
  IL: TList<Int64>;
  FL: TList<Double>;
begin
  FillChar(Result, SizeOf(Result), 0);
  case Typ of
    0: begin FStream.ReadBuffer(U8, 1); Result.Kind := gvInt; Result.I := U8; end;
    1: begin FStream.ReadBuffer(I8, 1); Result.Kind := gvInt; Result.I := I8; end;
    2: begin FStream.ReadBuffer(U16, 2); Result.Kind := gvInt; Result.I := U16; end;
    3: begin FStream.ReadBuffer(I16, 2); Result.Kind := gvInt; Result.I := I16; end;
    4: begin FStream.ReadBuffer(U32, 4); Result.Kind := gvInt; Result.I := U32; end;
    5: begin FStream.ReadBuffer(I32, 4); Result.Kind := gvInt; Result.I := I32; end;
    6: begin FStream.ReadBuffer(F32, 4); Result.Kind := gvFloat; Result.F := F32; end;
    7: begin FStream.ReadBuffer(B8, 1); Result.Kind := gvBool; Result.B := B8 <> 0; end;
    8: begin Result.Kind := gvStr; Result.S := ReadStr; end;
    10: begin FStream.ReadBuffer(U64, 8); Result.Kind := gvInt; Result.I := Int64(U64); end;
    11: begin FStream.ReadBuffer(I64, 8); Result.Kind := gvInt; Result.I := I64; end;
    12: begin FStream.ReadBuffer(F64, 8); Result.Kind := gvFloat; Result.F := F64; end;
    9: // Array
      begin
        ElemTyp := ReadU32;
        Count := ReadU64;
        if ElemTyp = 8 then
        begin
          SL := TList<string>.Create;
          try
            for N := 1 to Count do
              SL.Add(ReadStr);
            Result.Kind := gvStrArr;
            Result.SA := SL.ToArray;
          finally
            SL.Free;
          end;
        end
        else if ElemTyp in [6, 12] then
        begin
          FL := TList<Double>.Create;
          try
            for N := 1 to Count do
            begin
              Elem := ReadValue(ElemTyp);
              FL.Add(Elem.F);
            end;
            Result.Kind := gvFltArr;
            Result.FA := FL.ToArray;
          finally
            FL.Free;
          end;
        end
        else
        begin
          IL := TList<Int64>.Create;
          try
            for N := 1 to Count do
            begin
              Elem := ReadValue(ElemTyp);
              IL.Add(Elem.I);
            end;
            Result.Kind := gvIntArr;
            Result.IA := IL.ToArray;
          finally
            IL.Free;
          end;
        end;
      end;
  else
    raise Exception.CreateFmt('GGUF: unbekannter Werttyp %d', [Typ]);
  end;
end;

function TGgufFile.HasKey(const Key: string): Boolean;
begin
  Result := FMeta.ContainsKey(Key);
end;

function TGgufFile.MetaInt(const Key: string; Def: Int64): Int64;
var
  V: TGgufValue;
begin
  if FMeta.TryGetValue(Key, V) and (V.Kind = gvInt) then
    Result := V.I
  else
    Result := Def;
end;

function TGgufFile.MetaFloat(const Key: string; Def: Double): Double;
var
  V: TGgufValue;
begin
  if FMeta.TryGetValue(Key, V) and (V.Kind = gvFloat) then
    Result := V.F
  else if FMeta.TryGetValue(Key, V) and (V.Kind = gvInt) then
    Result := V.I
  else
    Result := Def;
end;

function TGgufFile.MetaBool(const Key: string; Def: Boolean): Boolean;
var
  V: TGgufValue;
begin
  if FMeta.TryGetValue(Key, V) and (V.Kind = gvBool) then
    Result := V.B
  else
    Result := Def;
end;

function TGgufFile.MetaStr(const Key, Def: string): string;
var
  V: TGgufValue;
begin
  if FMeta.TryGetValue(Key, V) and (V.Kind = gvStr) then
    Result := V.S
  else
    Result := Def;
end;

function TGgufFile.MetaStrArr(const Key: string): TArray<string>;
var
  V: TGgufValue;
begin
  if FMeta.TryGetValue(Key, V) and (V.Kind = gvStrArr) then
    Result := V.SA
  else
    Result := nil;
end;

function TGgufFile.MetaIntArr(const Key: string): TArray<Int64>;
var
  V: TGgufValue;
begin
  if FMeta.TryGetValue(Key, V) and (V.Kind = gvIntArr) then
    Result := V.IA
  else
    Result := nil;
end;

function TGgufFile.MetaFltArr(const Key: string): TArray<Double>;
var
  V: TGgufValue;
begin
  if FMeta.TryGetValue(Key, V) and (V.Kind = gvFltArr) then
    Result := V.FA
  else
    Result := nil;
end;

function TGgufFile.HasTensor(const Name: string): Boolean;
begin
  Result := FTensors.ContainsKey(Name);
end;

function TGgufFile.TensorInfo(const Name: string): TGgufTensorInfo;
begin
  if not FTensors.TryGetValue(Name, Result) then
    raise Exception.Create('GGUF: Tensor fehlt: ' + Name);
end;

function TGgufFile.LoadTensor(const Name: string): TQTensor;
var
  Info: TGgufTensorInfo;
  Bytes: Int64;
begin
  Info := TensorInfo(Name);
  if Integer(Info.Typ) = -1 then
    raise Exception.CreateFmt(
      'GGUF: Tensor "%s" hat einen nicht unterstuetzten Typ. ' +
      'Bitte Modell nach Q8_0, Q4_0 oder F16 requantisieren ' +
      '(llama-quantize model.gguf out.gguf Q8_0).', [Name]);
  if (Info.Rows > High(Integer)) or (Info.Cols > High(Integer)) then
    raise Exception.Create('GGUF: Tensordimension > 2^31: ' + Name);
  Result.Typ := Info.Typ;
  Result.Rows := Integer(Info.Rows);
  Result.Cols := Integer(Info.Cols);
  Bytes := Result.TotalBytes;
  SetLength(Result.Data, Bytes);
  FLock.Enter;
  try
    FStream.Position := FDataOffset + Info.Offset;
    FStream.ReadBuffer(Result.Data[0], Bytes);
  finally
    FLock.Leave;
  end;
end;

function TGgufFile.LoadTensorF32(const Name: string): TArray<Single>;
var
  T: TQTensor;
  R: Integer;
begin
  T := LoadTensor(Name);
  SetLength(Result, Int64(T.Rows) * T.Cols);
  for R := 0 to T.Rows - 1 do
    T.DequantRow(R, @Result[Int64(R) * T.Cols]);
end;

{ TGgufTokenizerBase }

constructor TGgufTokenizerBase.Create(Gg: TGgufFile);
var
  TypesArr: TArray<Int64>;
  I: Integer;
begin
  inherited Create;
  FStopIds := TList<Integer>.Create;
  FVocab := TDictionary<string, Integer>.Create;
  FPieces := Gg.MetaStrArr('tokenizer.ggml.tokens');
  if Length(FPieces) = 0 then
    raise Exception.Create('GGUF: tokenizer.ggml.tokens fehlt.');
  TypesArr := Gg.MetaIntArr('tokenizer.ggml.token_type');
  SetLength(FTypes, Length(FPieces));
  for I := 0 to High(FPieces) do
  begin
    if I <= High(TypesArr) then
      FTypes[I] := Integer(TypesArr[I])
    else
      FTypes[I] := 1;
    FVocab.AddOrSetValue(FPieces[I], I);
  end;
  FBos := Integer(Gg.MetaInt('tokenizer.ggml.bos_token_id', -1));
  FEos := Integer(Gg.MetaInt('tokenizer.ggml.eos_token_id', -1));
  FAddBos := Gg.MetaBool('tokenizer.ggml.add_bos_token', True);
  if FEos >= 0 then
    FStopIds.Add(FEos);
  DetectTemplate;
end;

destructor TGgufTokenizerBase.Destroy;
begin
  FVocab.Free;
  FStopIds.Free;
  inherited;
end;

function TGgufTokenizerBase.VocabSize: Integer;
begin
  Result := Length(FPieces);
end;

function TGgufTokenizerBase.BosId: Integer;
begin
  Result := FBos;
end;

function TGgufTokenizerBase.EosId: Integer;
begin
  Result := FEos;
end;

function TGgufTokenizerBase.IsStopToken(Id: Integer): Boolean;
begin
  Result := FStopIds.Contains(Id);
end;

function TGgufTokenizerBase.PrependBos: Boolean;
begin
  Result := FAddBos and (FBos >= 0);
end;

function TGgufTokenizerBase.PieceId(const Piece: string): Integer;
begin
  if not FVocab.TryGetValue(Piece, Result) then
    Result := -1;
end;

procedure TGgufTokenizerBase.DetectTemplate;
var
  ImEnd: Integer;
begin
  if PieceId('<|im_start|>') >= 0 then
  begin
    FAutoTemplate := ctChatML;
    ImEnd := PieceId('<|im_end|>');
    if (ImEnd >= 0) and not FStopIds.Contains(ImEnd) then
      FStopIds.Add(ImEnd);
  end
  else if Self is TSpmTokenizer then
    FAutoTemplate := ctLlama2
  else
    FAutoTemplate := ctPlain;
end;

procedure TGgufTokenizerBase.AppendChat(Res: TList<Integer>;
  const Msgs: TChatMessages; Template: TChatTemplate);
var
  M: TChatMessage;
  SysText, UserBuf, T: string;
  ImStart, ImEnd: Integer;
  First: Boolean;

  procedure AppendText(const S: string);
  begin
    if S <> '' then
      Res.AddRange(Encode(S));
  end;

begin
  if Template = ctAuto then
    Template := FAutoTemplate;
  if FAddBos and (FBos >= 0) then
    Res.Add(FBos);

  case Template of
    ctChatML:
      begin
        ImStart := PieceId('<|im_start|>');
        ImEnd := PieceId('<|im_end|>');
        for M in Msgs do
        begin
          Res.Add(ImStart);
          AppendText(M.Role + #10 + M.Content);
          Res.Add(ImEnd);
          AppendText(#10);
        end;
        Res.Add(ImStart);
        AppendText('assistant' + #10);
      end;
    ctLlama2:
      begin
        SysText := '';
        UserBuf := '';
        T := '';
        First := True;
        for M in Msgs do
        begin
          if SameText(M.Role, 'system') then
            SysText := M.Content
          else if SameText(M.Role, 'user') then
            UserBuf := M.Content
          else if SameText(M.Role, 'assistant') then
          begin
            if First and (SysText <> '') then
              T := T + '[INST] <<SYS>>'#10 + SysText + #10'<</SYS>>'#10#10 +
                UserBuf + ' [/INST] '
            else
              T := T + '[INST] ' + UserBuf + ' [/INST] ';
            T := T + M.Content + ' ';
            UserBuf := '';
            First := False;
          end;
        end;
        if First and (SysText <> '') then
          T := T + '[INST] <<SYS>>'#10 + SysText + #10'<</SYS>>'#10#10 +
            UserBuf + ' [/INST]'
        else
          T := T + '[INST] ' + UserBuf + ' [/INST]';
        AppendText(T);
      end;
  else // ctPlain und alles andere
    begin
      T := '';
      for M in Msgs do
      begin
        if SameText(M.Role, 'system') then
          T := T + '### System:'#10 + M.Content + #10#10
        else if SameText(M.Role, 'assistant') then
          T := T + '### Assistant:'#10 + M.Content + #10#10
        else
          T := T + '### User:'#10 + M.Content + #10#10;
      end;
      T := T + '### Assistant:'#10;
      AppendText(T);
    end;
  end;
end;

function TGgufTokenizerBase.BuildChatTokens(const Msgs: TChatMessages;
  Template: TChatTemplate): TArray<Integer>;
var
  Res: TList<Integer>;
begin
  Res := TList<Integer>.Create;
  try
    AppendChat(Res, Msgs, Template);
    Result := Res.ToArray;
  finally
    Res.Free;
  end;
end;

{ TSpmTokenizer }

constructor TSpmTokenizer.Create(Gg: TGgufFile);
var
  ScoresArr: TArray<Double>;
  I, B: Integer;
begin
  inherited Create(Gg);
  ScoresArr := Gg.MetaFltArr('tokenizer.ggml.scores');
  SetLength(FScores, Length(FPieces));
  for I := 0 to High(FPieces) do
    if I <= High(ScoresArr) then
      FScores[I] := ScoresArr[I]
    else
      FScores[I] := 0;
  FAddSpacePrefix := Gg.MetaBool('tokenizer.ggml.add_space_prefix', True);
  for B := 0 to 255 do
    FByteTok[B] := PieceId(Format('<0x%.2X>', [B]));
end;

function TSpmTokenizer.Encode(const Text: string): TArray<Integer>;
var
  S: string;
  Syms: TList<Integer>;       // Token-Ids der aktuellen Symbolfolge
  Pieces: TList<string>;      // zugehoerige Strings
  I, J, Id, BestI, MergedId: Integer;
  CP: string;
  BestScore: Single;
  Merged: string;
  Utf8: TBytes;
begin
  Syms := TList<Integer>.Create;
  Pieces := TList<string>.Create;
  try
    S := Text;
    if FAddSpacePrefix and (S <> '') then
      S := ' ' + S;
    S := StringReplace(S, ' ', #$2581, [rfReplaceAll]);

    { Startzustand: ein Symbol pro Codepoint, Byte-Fallback fuer Unbekanntes }
    I := 1;
    while I <= Length(S) do
    begin
      if (I < Length(S)) and Char.IsSurrogatePair(S, I) then
      begin
        CP := Copy(S, I, 2);
        Inc(I, 2);
      end
      else
      begin
        CP := S[I];
        Inc(I);
      end;
      Id := PieceId(CP);
      if Id >= 0 then
      begin
        Syms.Add(Id);
        Pieces.Add(CP);
      end
      else
      begin
        Utf8 := TEncoding.UTF8.GetBytes(CP);
        for J := 0 to High(Utf8) do
          if FByteTok[Utf8[J]] >= 0 then
          begin
            Syms.Add(FByteTok[Utf8[J]]);
            Pieces.Add(FPieces[FByteTok[Utf8[J]]]);
          end;
      end;
    end;

    { Greedy-Merge nach Score (SentencePiece-BPE) }
    while Syms.Count > 1 do
    begin
      BestI := -1;
      BestScore := -1e30;
      MergedId := -1;
      for I := 0 to Syms.Count - 2 do
      begin
        Merged := Pieces[I] + Pieces[I + 1];
        Id := PieceId(Merged);
        if (Id >= 0) and (FScores[Id] > BestScore) then
        begin
          BestScore := FScores[Id];
          BestI := I;
          MergedId := Id;
        end;
      end;
      if BestI < 0 then
        Break;
      Pieces[BestI] := Pieces[BestI] + Pieces[BestI + 1];
      Syms[BestI] := MergedId;
      Pieces.Delete(BestI + 1);
      Syms.Delete(BestI + 1);
    end;
    Result := Syms.ToArray;
  finally
    Syms.Free;
    Pieces.Free;
  end;
end;

function TSpmTokenizer.TokenBytes(Id: Integer): TBytes;
var
  P: string;
  B: Integer;
begin
  if (Id < 0) or (Id > High(FPieces)) then
    Exit(nil);
  if FTypes[Id] = TT_CONTROL then
    Exit(nil);
  P := FPieces[Id];
  if (FTypes[Id] = TT_BYTE) or ((Length(P) = 6) and P.StartsWith('<0x') and
    P.EndsWith('>')) then
  begin
    B := StrToIntDef('$' + Copy(P, 4, 2), -1);
    if B >= 0 then
    begin
      SetLength(Result, 1);
      Result[0] := Byte(B);
      Exit;
    end;
  end;
  P := StringReplace(P, #$2581, ' ', [rfReplaceAll]);
  Result := TEncoding.UTF8.GetBytes(P);
end;

{ TGpt2Tokenizer }

constructor TGpt2Tokenizer.Create(Gg: TGgufFile);
var
  Merges: TArray<string>;
  I, N, B: Integer;
begin
  inherited Create(Gg);
  FMergeRank := TDictionary<string, Integer>.Create;
  FCharToByte := TDictionary<Char, Byte>.Create;
  Merges := Gg.MetaStrArr('tokenizer.ggml.merges');
  for I := 0 to High(Merges) do
    FMergeRank.AddOrSetValue(Merges[I], I);
  { GPT-2 bytes_to_unicode: druckbare Bytes bleiben, Rest wird ab U+0100
    durchnummeriert }
  N := 0;
  for B := 0 to 255 do
  begin
    if ((B >= 33) and (B <= 126)) or ((B >= 161) and (B <= 172)) or
      ((B >= 174) and (B <= 255)) then
      FByteToChar[B] := Char(B)
    else
    begin
      FByteToChar[B] := Char(256 + N);
      Inc(N);
    end;
    FCharToByte.AddOrSetValue(FByteToChar[B], Byte(B));
  end;
  FAddBos := Gg.MetaBool('tokenizer.ggml.add_bos_token', False);
end;

destructor TGpt2Tokenizer.Destroy;
begin
  FMergeRank.Free;
  FCharToByte.Free;
  inherited;
end;

procedure TGpt2Tokenizer.EncodeChunk(const Chunk: string; Res: TList<Integer>);
var
  Syms: TList<string>;
  Utf8: TBytes;
  I, BestI, BestRank, R, Id: Integer;
  SB: TStringBuilder;
begin
  if Chunk = '' then
    Exit;
  Utf8 := TEncoding.UTF8.GetBytes(Chunk);
  Syms := TList<string>.Create;
  SB := TStringBuilder.Create;
  try
    for I := 0 to High(Utf8) do
      Syms.Add(FByteToChar[Utf8[I]]);
    { Merges nach Rang anwenden }
    while Syms.Count > 1 do
    begin
      BestI := -1;
      BestRank := MaxInt;
      for I := 0 to Syms.Count - 2 do
        if FMergeRank.TryGetValue(Syms[I] + ' ' + Syms[I + 1], R) and
          (R < BestRank) then
        begin
          BestRank := R;
          BestI := I;
        end;
      if BestI < 0 then
        Break;
      Syms[BestI] := Syms[BestI] + Syms[BestI + 1];
      Syms.Delete(BestI + 1);
    end;
    for I := 0 to Syms.Count - 1 do
    begin
      Id := PieceId(Syms[I]);
      if Id >= 0 then
        Res.Add(Id)
      else
      begin
        { Symbol nicht im Vokabular -> in Einzelzeichen zerlegen }
        SB.Clear;
        SB.Append(Syms[I]);
        for R := 0 to SB.Length - 1 do
        begin
          Id := PieceId(SB.Chars[R]);
          if Id >= 0 then
            Res.Add(Id);
        end;
      end;
    end;
  finally
    Syms.Free;
    SB.Free;
  end;
end;

function TGpt2Tokenizer.Encode(const Text: string): TArray<Integer>;
var
  Res: TList<Integer>;
  I, Start: Integer;

  function Cat(C: Char): Integer;
  begin
    if C.IsLetter then
      Result := 0
    else if C.IsDigit then
      Result := 1
    else if C = ' ' then
      Result := 2
    else
      Result := 3;
  end;

var
  CurCat: Integer;
  Chunk: string;
begin
  { Vereinfachter GPT-2-Pretokenizer: an Kategorie-Grenzen splitten,
    fuehrendes Leerzeichen klebt am Folgewort (" word"-Konvention). }
  Res := TList<Integer>.Create;
  try
    I := 1;
    while I <= Length(Text) do
    begin
      Start := I;
      if (Text[I] = ' ') and (I < Length(Text)) and (Text[I + 1] <> ' ') then
        Inc(I); // Leerzeichen dem naechsten Chunk zuschlagen
      if I <= Length(Text) then
      begin
        CurCat := Cat(Text[I]);
        while (I <= Length(Text)) and (Cat(Text[I]) = CurCat) and
          (Text[I] <> ' ') do
          Inc(I);
        if (CurCat = 2) then // reine Leerzeichen-Folge
          while (I <= Length(Text)) and (Text[I] = ' ') do
            Inc(I);
      end;
      Chunk := Copy(Text, Start, I - Start);
      EncodeChunk(Chunk, Res);
    end;
    Result := Res.ToArray;
  finally
    Res.Free;
  end;
end;

function TGpt2Tokenizer.TokenBytes(Id: Integer): TBytes;
var
  P: string;
  I, N: Integer;
  B: Byte;
begin
  if (Id < 0) or (Id > High(FPieces)) then
    Exit(nil);
  if FTypes[Id] = TT_CONTROL then
    Exit(nil);
  P := FPieces[Id];
  SetLength(Result, Length(P) * 4);
  N := 0;
  for I := 1 to Length(P) do
    if FCharToByte.TryGetValue(P[I], B) then
    begin
      Result[N] := B;
      Inc(N);
    end
    else
    begin
      { Zeichen ausserhalb der Byte-Map: als UTF-8 uebernehmen }
      var Enc := TEncoding.UTF8.GetBytes(string(P[I]));
      Move(Enc[0], Result[N], Length(Enc));
      Inc(N, Length(Enc));
    end;
  SetLength(Result, N);
end;

function CreateGgufTokenizer(Gg: TGgufFile): TGgufTokenizerBase;
var
  Model: string;
begin
  Model := Gg.MetaStr('tokenizer.ggml.model', 'llama');
  if SameText(Model, 'gpt2') then
    Result := TGpt2Tokenizer.Create(Gg)
  else if SameText(Model, 'llama') then
    Result := TSpmTokenizer.Create(Gg)
  else
    raise Exception.CreateFmt('GGUF: Tokenizer-Modell "%s" nicht unterstuetzt ' +
      '(llama/gpt2).', [Model]);
end;

end.
