unit Prism.Multimodal;

{ Multimodal training data pipeline.

  Approach: byte-level universality. The Prism tokenizer operates on
  bytes, so ANY kind of data can be tokenized: text, images, audio,
  video, 3D data, arbitrary binary data. Modalities are framed by
  special token markers (<|img|>...<|/img|> etc.), so the model learns
  to distinguish the data type from context.

  Large raw data is reduced to a maximum size via decimation
  (stride sampling). For serious image/audio quality, learned encoders
  (patch/mel embeddings) are the next step (roadmap); the interface
  here stays the same. }

interface

uses
  System.SysUtils, System.Classes, System.SyncObjs, Prism.Tokenizer;

type
  TModality = (mText, mImage, mAudio, mVideo, m3D, mBinary);

function ModalityFromString(const S: string): TModality;
function ModalityOpenTag(M: TModality): string;
function ModalityCloseTag(M: TModality): string;

{ Reduces Data via stride sampling to at most MaxBytes }
function DownsampleBytes(const Data: TBytes; MaxBytes: Integer): TBytes;

{ Chat training sample: <|user|>question<|assistant|>answer<|eos|> }
function MakeTextSample(const UserText, AssistantText: string): TBytes;

{ Multimodal sample: raw data + description as target answer }
function MakeDataSample(const Data: TBytes; M: TModality;
  const Description: string; MaxBytes: Integer = 65536): TBytes;

{ Thread-safe appending to the corpus file }
procedure AppendToCorpus(const CorpusPath: string; const Sample: TBytes);

implementation

var
  GCorpusLock: TCriticalSection;

function ModalityFromString(const S: string): TModality;
begin
  if SameText(S, 'image') or SameText(S, 'img') then
    Result := mImage
  else if SameText(S, 'audio') then
    Result := mAudio
  else if SameText(S, 'video') then
    Result := mVideo
  else if SameText(S, '3d') then
    Result := m3D
  else if SameText(S, 'binary') or SameText(S, 'bin') then
    Result := mBinary
  else
    Result := mText;
end;

function ModalityOpenTag(M: TModality): string;
begin
  case M of
    mImage:  Result := '<|img|>';
    mAudio:  Result := '<|aud|>';
    mVideo:  Result := '<|vid|>';
    m3D:     Result := '<|3d|>';
    mBinary: Result := '<|bin|>';
  else
    Result := '';
  end;
end;

function ModalityCloseTag(M: TModality): string;
begin
  case M of
    mImage:  Result := '<|/img|>';
    mAudio:  Result := '<|/aud|>';
    mVideo:  Result := '<|/vid|>';
    m3D:     Result := '<|/3d|>';
    mBinary: Result := '<|/bin|>';
  else
    Result := '';
  end;
end;

function DownsampleBytes(const Data: TBytes; MaxBytes: Integer): TBytes;
var
  Stride: Double;
  I: Integer;
begin
  if (MaxBytes <= 0) or (Length(Data) <= MaxBytes) then
    Exit(Data);
  SetLength(Result, MaxBytes);
  Stride := Length(Data) / MaxBytes;
  for I := 0 to MaxBytes - 1 do
    Result[I] := Data[Trunc(I * Stride)];
end;

function Concat(const Parts: array of TBytes): TBytes;
var
  I, N, P: Integer;
begin
  N := 0;
  for I := 0 to High(Parts) do
    Inc(N, Length(Parts[I]));
  SetLength(Result, N);
  P := 0;
  for I := 0 to High(Parts) do
    if Length(Parts[I]) > 0 then
    begin
      Move(Parts[I][0], Result[P], Length(Parts[I]));
      Inc(P, Length(Parts[I]));
    end;
end;

function U8(const S: string): TBytes;
begin
  Result := TEncoding.UTF8.GetBytes(S);
end;

function MakeTextSample(const UserText, AssistantText: string): TBytes;
begin
  Result := Concat([U8('<|user|>'), U8(UserText), U8('<|assistant|>'),
    U8(AssistantText), U8('<|eos|>'#10)]);
end;

function MakeDataSample(const Data: TBytes; M: TModality;
  const Description: string; MaxBytes: Integer): TBytes;
var
  D: TBytes;
begin
  D := DownsampleBytes(Data, MaxBytes);
  Result := Concat([U8('<|user|>'), U8(ModalityOpenTag(M)), D,
    U8(ModalityCloseTag(M)), U8('<|assistant|>'), U8(Description),
    U8('<|eos|>'#10)]);
end;

procedure AppendToCorpus(const CorpusPath: string; const Sample: TBytes);
var
  FS: TFileStream;
begin
  if Length(Sample) = 0 then
    Exit;
  GCorpusLock.Enter;
  try
    if FileExists(CorpusPath) then
      FS := TFileStream.Create(CorpusPath, fmOpenReadWrite or fmShareDenyWrite)
    else
      FS := TFileStream.Create(CorpusPath, fmCreate);
    try
      FS.Seek(0, soEnd);
      FS.WriteBuffer(Sample[0], Length(Sample));
    finally
      FS.Free;
    end;
  finally
    GCorpusLock.Leave;
  end;
end;

initialization
  GCorpusLock := TCriticalSection.Create;

finalization
  GCorpusLock.Free;

end.
