program PrismMobile;

{ Prism Mobile - lokaler LLM-Server als FMX-App (Android/iOS/Win/macOS).
  Projekt in der IDE oeffnen, Zielplattform waehlen (z.B. Android 64-Bit),
  Modell-Dateien im Deployment-Manager hinterlegen, deployen. }

uses
  System.StartUpCopy,
  FMX.Forms,
  Prism.Types in '..\..\src\Prism.Types.pas',
  Prism.Vector in '..\..\src\Prism.Vector.pas',
  Prism.Tensor in '..\..\src\Prism.Tensor.pas',
  Prism.Tokenizer in '..\..\src\Prism.Tokenizer.pas',
  Prism.Model in '..\..\src\Prism.Model.pas',
  Prism.Streaming in '..\..\src\Prism.Streaming.pas',
  Prism.Gpu in '..\..\src\Prism.Gpu.pas',
  Prism.Inference in '..\..\src\Prism.Inference.pas',
  Prism.Gguf in '..\..\src\Prism.Gguf.pas',
  Prism.Llama in '..\..\src\Prism.Llama.pas',
  Prism.Train in '..\..\src\Prism.Train.pas',
  Prism.Multimodal in '..\..\src\Prism.Multimodal.pas',
  Prism.Verify in '..\..\src\Prism.Verify.pas',
  Prism.RestServer in '..\..\src\Prism.RestServer.pas',
  MainFormU in 'MainFormU.pas' {MainForm};

{$R *.res}

begin
  Application.Initialize;
  Application.CreateForm(TMainForm, MainForm);
  Application.Run;
end.
