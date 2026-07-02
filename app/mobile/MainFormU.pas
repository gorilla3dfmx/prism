unit MainFormU;

{ Prism Mobile: starts the Prism LLM server locally on the device (FMX,
  Android/iOS/Windows/macOS). The UI is built at runtime so that the
  .fmx file stays minimal.

  Model deployment:
  - Android/iOS: place model.prism + tokenizer.json (or model.gguf) into
    the app's documents directory via the Deployment Manager
    (TPath.GetDocumentsPath), remote path ".\" or "StartUp\Documents".
  - Android requires the INTERNET permission (project options).
  - For billion-parameter GGUF models use --stream-layers
    (the "Layer-Cache" slider). }

interface

uses
  System.SysUtils, System.Classes, System.Types, System.IOUtils,
  System.UITypes,
  FMX.Types, FMX.Forms, FMX.Controls, FMX.StdCtrls, FMX.Edit, FMX.Memo,
  FMX.Layouts, FMX.Controls.Presentation, FMX.ScrollBox, FMX.Memo.Types,
  Prism.Types, Prism.RestServer;

type
  TMainForm = class(TForm)
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
  private
    FToolbar: TToolBar;
    FTitle: TLabel;
    FMemo: TMemo;
    FBottom: TLayout;
    FPortEdit: TEdit;
    FStreamEdit: TEdit;
    FStartBtn: TButton;
    FServer: TPrismRestServer;
    procedure StartStopClick(Sender: TObject);
    procedure AddLog(const S: string);
    function FindModelFile(out ModelPath, TokPath: string): Boolean;
  end;

var
  MainForm: TMainForm;

implementation

{$R *.fmx}

procedure TMainForm.FormCreate(Sender: TObject);
begin
  Caption := 'Prism LLM';

  FToolbar := TToolBar.Create(Self);
  FToolbar.Parent := Self;
  FTitle := TLabel.Create(Self);
  FTitle.Parent := FToolbar;
  FTitle.Align := TAlignLayout.Client;
  FTitle.TextSettings.HorzAlign := TTextAlign.Center;
  FTitle.Text := 'Prism ' + PRISM_VERSION + ' - local LLM server';

  FBottom := TLayout.Create(Self);
  FBottom.Parent := Self;
  FBottom.Align := TAlignLayout.Bottom;
  FBottom.Height := 56;
  FBottom.Padding.Rect := TRectF.Create(8, 8, 8, 8);

  FPortEdit := TEdit.Create(Self);
  FPortEdit.Parent := FBottom;
  FPortEdit.Align := TAlignLayout.Left;
  FPortEdit.Width := 90;
  FPortEdit.Text := '11434';
  FPortEdit.TextPrompt := 'Port';

  FStreamEdit := TEdit.Create(Self);
  FStreamEdit.Parent := FBottom;
  FStreamEdit.Align := TAlignLayout.Left;
  FStreamEdit.Width := 90;
  FStreamEdit.Margins.Left := 8;
  FStreamEdit.Text := '4';
  FStreamEdit.TextPrompt := 'Layer-Cache';

  FStartBtn := TButton.Create(Self);
  FStartBtn.Parent := FBottom;
  FStartBtn.Align := TAlignLayout.Client;
  FStartBtn.Margins.Left := 8;
  FStartBtn.Text := 'Start server';
  FStartBtn.OnClick := StartStopClick;

  FMemo := TMemo.Create(Self);
  FMemo.Parent := Self;
  FMemo.Align := TAlignLayout.Client;
  FMemo.ReadOnly := True;
  FMemo.TextSettings.Font.Family := 'Consolas';

  AddLog('Ready. Place the model in the documents directory:');
  AddLog('  ' + TPath.GetDocumentsPath);
  AddLog('Looking for: model.prism + tokenizer.json or *.gguf');
end;

procedure TMainForm.FormDestroy(Sender: TObject);
begin
  FServer.Free;
end;

procedure TMainForm.AddLog(const S: string);
begin
  if TThread.CurrentThread.ThreadID <> MainThreadID then
    TThread.Queue(nil,
      procedure
      begin
        FMemo.Lines.Add(S);
        FMemo.GoToTextEnd;
      end)
  else
  begin
    FMemo.Lines.Add(S);
    FMemo.GoToTextEnd;
  end;
end;

function TMainForm.FindModelFile(out ModelPath, TokPath: string): Boolean;
var
  Dir, F: string;
  Files: TArray<string>;
begin
  Result := False;
  ModelPath := '';
  TokPath := '';
  Dir := TPath.GetDocumentsPath;
  if TFile.Exists(TPath.Combine(Dir, 'model.prism')) then
  begin
    ModelPath := TPath.Combine(Dir, 'model.prism');
    TokPath := TPath.Combine(Dir, 'tokenizer.json');
    Exit(TFile.Exists(TokPath));
  end;
  Files := TDirectory.GetFiles(Dir, '*.gguf');
  for F in Files do
  begin
    ModelPath := F;
    Exit(True);
  end;
end;

procedure TMainForm.StartStopClick(Sender: TObject);
var
  Opts: TServerOptions;
  ModelPath, TokPath: string;
begin
  if FServer <> nil then
  begin
    FreeAndNil(FServer);
    FStartBtn.Text := 'Start server';
    AddLog('Server stopped.');
    Exit;
  end;
  if not FindModelFile(ModelPath, TokPath) then
  begin
    AddLog('ERROR: no model found in ' + TPath.GetDocumentsPath);
    Exit;
  end;
  FStartBtn.Enabled := False;
  AddLog('Loading model (please wait) ...');
  TThread.CreateAnonymousThread(
    procedure
    var
      Srv: TPrismRestServer;
    begin
      try
        Opts := TServerOptions.Default;
        Opts.ModelPath := ModelPath;
        Opts.TokenizerPath := TokPath;
        Opts.Port := StrToIntDef(FPortEdit.Text, 11434);
        Opts.StreamLayers := StrToIntDef(FStreamEdit.Text, 4);
        Opts.CtxOverride := 1024; // limit the KV cache on mobile devices
        Opts.CorpusPath := TPath.Combine(TPath.GetDocumentsPath, 'corpus.bin');
        Srv := TPrismRestServer.Create(Opts,
          procedure(S: string)
          begin
            AddLog(S);
          end);
        Srv.Start;
        TThread.Queue(nil,
          procedure
          begin
            FServer := Srv;
            FStartBtn.Text := 'Stop server';
            FStartBtn.Enabled := True;
          end);
      except
        on E: Exception do
        begin
          AddLog('ERROR: ' + E.Message);
          TThread.Queue(nil,
            procedure
            begin
              FStartBtn.Enabled := True;
            end);
        end;
      end;
    end).Start;
end;

end.
