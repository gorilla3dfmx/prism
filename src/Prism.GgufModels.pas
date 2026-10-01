unit Prism.GgufModels;

{ Opens a GGUF file and picks the engine from general.architecture:
  gemma4 -> Prism.Gemma4, everything else -> Prism.Llama (Llama, Mistral,
  Qwen2/3, Phi, ...). Callers only ever see TGgufBackend, so adding an
  architecture means one more branch here and nothing in the REST server. }

interface

uses
  System.SysUtils, Prism.Gguf;

function CreateGgufBackend(const Path: string; CtxOverride: Integer;
  StreamLayers: Integer; const Log: TProc<string>): TGgufBackend;

implementation

uses
  Prism.Llama, Prism.Gemma4;

function CreateGgufBackend(const Path: string; CtxOverride: Integer;
  StreamLayers: Integer; const Log: TProc<string>): TGgufBackend;
var
  Gg: TGgufFile;
begin
  { From here on Gg belongs to the backend: its model stores the file first
    thing and frees it again if anything later in construction fails. }
  Gg := TGgufFile.Create(Path);
  if SameText(Gg.Arch, 'gemma4') then
    Result := TGemma4Backend.Create(Gg, CtxOverride, StreamLayers, Log)
  else
    Result := TLlamaBackend.Create(Gg, CtxOverride, StreamLayers, Log);
end;

end.
