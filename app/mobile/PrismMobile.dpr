program PrismMobile;

{ Prism Mobile - local LLM server as an FMX app (Android/iOS/Win/macOS).
  Open the project in the IDE, choose the target platform (e.g. Android 64-bit),
  add the model files in the Deployment Manager, then deploy. }

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
  Prism.Laws in '..\..\src\Prism.Laws.pas',
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
