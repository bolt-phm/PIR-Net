#nullable disable
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using System.Linq;

namespace BoltDetectionGUI
{
    public partial class Form1 : Form
    {
        // ================= 1. Global Config & Paths =================

        string projectDir = "";
        string configFileName = "config.json";
        string inferenceScript = "inference_engine.py";
        string logImgName = "final_generalization_matrix.png";

        // Dynamic Paths
        string ConfigPath => string.IsNullOrEmpty(projectDir) ? "" : Path.Combine(projectDir, configFileName);
        string InferencePath => string.IsNullOrEmpty(projectDir) ? "" : Path.Combine(projectDir, inferenceScript);

        // Get Absolute Image Path
        string LogImgFullPath
        {
            get
            {
                if (string.IsNullOrEmpty(projectDir) || txtLogDir == null) return "";
                string dir = string.IsNullOrWhiteSpace(txtLogDir.Text) ? "tf-logs" : txtLogDir.Text;
                return Path.GetFullPath(Path.Combine(projectDir, dir, logImgName));
            }
        }

        // ================= 2. Control Declarations =================

        // --- Top Container ---
        private TextBox txtProjectPath;
        private TextBox txtPythonPath;
        private Button btnBrowse, btnBrowsePy;

        // --- Main Container ---
        private TabControl mainTabControl;
        private TabControl configTabControl;
        private RichTextBox rtbLog;
        private PictureBox picResult;

        // --- Core Buttons ---
        private Button btnSave;

        // --- [Tab 1: Training] ---
        private NumericUpDown numEpochs, numWarmup;
        private TextBox txtLR, txtWeightDecay;
        private TextBox txtLogDir;
        private ComboBox cmbOptimizer, cmbDevice, cmbLossType;

        // --- [Tab 2: Data] ---
        private NumericUpDown numBatchSize, numWorkers;
        private TextBox txtDataDir;
        private CheckBox chkUseWhitelist;
        private CheckedListBox chkListFiles;

        // --- [Tab 3: Model & Ensemble] ---
        private CheckBox chkSigAug, chkImgAug;
        private NumericUpDown numClasses;

        // Specific Model Config
        private ComboBox cmbPseudoMode; // pseudo_image_mode
        private ComboBox cmbImgType;    // image_model -> type
        private ComboBox cmbSigType;    // signal_model -> type
        private ComboBox cmbFusionType; // fusion -> type

        // Ensemble List
        private ListBox lstEnsembleModels;
        private Button btnAddModel, btnRemoveModel;

        // --- [Tab 4: Project Info] ---
        private TextBox txtProjName, txtProjVersion, txtProjDesc;

        // --- [Tab 5: Execution] ---
        private RadioButton rbModeUSB;
        private RadioButton rbModeBatch;
        private CheckBox chkEnableEnsemble;
        private Panel panelModeUSB;
        private Panel panelModeBatch;

        private Button btnRunUSB;
        private Label lblUsbStatus;
        private Label lblPredClass;
        private Label lblConfidence;
        private ProgressBar probBar;
        private RichTextBox rtbUsbLog;
        private Button btnRunBatch;

        public Form1()
        {
            InitCustomGUI();
        }

        // ==========================================================
        // 3. UI Construction (Layout)
        // ==========================================================
        private void InitCustomGUI()
        {
            this.Text = "Bolt Detection System";
            this.Size = new Size(1280, 900); // Slightly reduced height for safety
            this.StartPosition = FormStartPosition.CenterScreen;
            this.Font = new Font("Segoe UI", 9F, FontStyle.Regular, GraphicsUnit.Point, 0); // Changed to Segoe UI for English
            this.BackColor = Color.FromArgb(245, 247, 250);

            // --- A. Top Panel ---
            Panel panelTop = new Panel() { Dock = DockStyle.Top, Height = 110, BackColor = Color.White, Padding = new Padding(15) };
            Panel borderLine = new Panel() { Dock = DockStyle.Bottom, Height = 1, BackColor = Color.LightGray };
            panelTop.Controls.Add(borderLine);

            // Row 1: Project Path
            Label lblPath = new Label() { Text = "Project Path:", Location = new Point(20, 28), AutoSize = true, Font = new Font("Segoe UI", 9, FontStyle.Bold) };
            txtProjectPath = new TextBox() { Location = new Point(150, 25), Width = 840, ReadOnly = true, BackColor = Color.WhiteSmoke, BorderStyle = BorderStyle.FixedSingle };
            btnBrowse = CreateStyledButton("📂 Browse...", new Point(1010, 23), 140, 30, Color.FromArgb(0, 120, 215), Color.White);
            btnBrowse.Click += BtnBrowse_Click;

            // Row 2: Python Path
            Label lblPy = new Label() { Text = "Python Env:", Location = new Point(20, 68), AutoSize = true, Font = new Font("Segoe UI", 9, FontStyle.Bold) };
            txtPythonPath = new TextBox() { Location = new Point(150, 65), Width = 840, ReadOnly = true, BackColor = Color.WhiteSmoke, BorderStyle = BorderStyle.FixedSingle, Text = "python" };
            btnBrowsePy = CreateStyledButton("🐍 Select Python", new Point(1010, 63), 140, 30, Color.Teal, Color.White);
            btnBrowsePy.Click += BtnBrowsePy_Click;

            // [FIXED] Removed duplicate Controls.AddRange call
            panelTop.Controls.AddRange(new Control[] { lblPath, txtProjectPath, btnBrowse, lblPy, txtPythonPath, btnBrowsePy });
            this.Controls.Add(panelTop);

            // --- B. Main TabControl ---
            mainTabControl = new TabControl() { Dock = DockStyle.Fill, ItemSize = new Size(150, 30), SizeMode = TabSizeMode.Fixed };
            TabPage pageConfigCenter = new TabPage("⚙️ Configuration");
            TabPage pageRunTest = new TabPage("🚀 Execution");

            pageConfigCenter.Padding = new Padding(10);
            pageConfigCenter.BackColor = Color.FromArgb(245, 247, 250);

            mainTabControl.Controls.Add(pageConfigCenter);
            mainTabControl.Controls.Add(pageRunTest);

            Panel panelMain = new Panel() { Dock = DockStyle.Fill, Padding = new Padding(10) };
            panelMain.Controls.Add(mainTabControl);
            this.Controls.Add(panelMain);
            panelTop.SendToBack();

            // =================================================
            // TAB 1: Configuration Center
            // =================================================
            configTabControl = new TabControl() { Dock = DockStyle.Fill };
            TabPage subTabTrain = new TabPage("Training");
            TabPage subTabData = new TabPage("Data");
            TabPage subTabModel = new TabPage("Model & Ensemble");
            TabPage subTabProject = new TabPage("Project Info");

            configTabControl.Controls.AddRange(new Control[] { subTabTrain, subTabData, subTabModel, subTabProject });

            // Bottom Save Button
            Panel panelConfigBottom = new Panel() { Dock = DockStyle.Bottom, Height = 70, Padding = new Padding(0, 15, 0, 10) };
            btnSave = CreateStyledButton("💾 Save Config to JSON", new Point(0, 0), 250, 45, Color.ForestGreen, Color.White);
            btnSave.Dock = DockStyle.Right;
            btnSave.Font = new Font("Segoe UI", 11, FontStyle.Bold);
            btnSave.Click += BtnSave_Click;
            panelConfigBottom.Controls.Add(btnSave);

            pageConfigCenter.Controls.Add(configTabControl);
            pageConfigCenter.Controls.Add(panelConfigBottom);

            // Fill Tabs
            BuildTrainTab(subTabTrain);
            BuildDataTab(subTabData);
            BuildModelTab(subTabModel);
            BuildProjectTab(subTabProject);

            // =================================================
            // TAB 2: Execution
            // =================================================
            BuildRunTab(pageRunTest);
        }

        private Button CreateStyledButton(string text, Point loc, int w, int h, Color bg, Color fg)
        {
            return new Button()
            {
                Text = text,
                Location = loc,
                Size = new Size(w, h),
                BackColor = bg,
                ForeColor = fg,
                FlatStyle = FlatStyle.Flat,
                Font = new Font("Segoe UI", 9, FontStyle.Bold),
                Cursor = Cursors.Hand,
                FlatAppearance = { BorderSize = 0 }
            };
        }

        private void BuildRunTab(TabPage page)
        {
            // 1. Main Panel
            Panel mainPanel = new Panel() { Dock = DockStyle.Fill, BackColor = Color.WhiteSmoke };
            page.Controls.Add(mainPanel);

            // 2. Mode Selection Group
            GroupBox grpMode = new GroupBox()
            {
                Text = "Test Mode Selection",
                Dock = DockStyle.Top,
                Height = 90,
                Padding = new Padding(10),
                Font = new Font("Segoe UI", 10, FontStyle.Bold)
            };
            rbModeUSB = new RadioButton() { Text = "🔌 USB Real-time Inference", Location = new Point(30, 40), AutoSize = true, Checked = true, ForeColor = Color.DarkSlateBlue };
            rbModeBatch = new RadioButton() { Text = "📊 Batch Generalization Test", Location = new Point(350, 40), AutoSize = true, ForeColor = Color.DarkSlateBlue };
            chkEnableEnsemble = new CheckBox() { Text = "🎲 Enable Ensemble", Location = new Point(700, 40), AutoSize = true, ForeColor = Color.Crimson };

            rbModeUSB.CheckedChanged += (s, e) => ToggleTestMode();
            grpMode.Controls.AddRange(new Control[] { rbModeUSB, rbModeBatch, chkEnableEnsemble });
            mainPanel.Controls.Add(grpMode);

            // 3. Container
            Panel container = new Panel() { Dock = DockStyle.Fill, Padding = new Padding(10) };
            mainPanel.Controls.Add(container);
            container.BringToFront();
            grpMode.SendToBack();

            // ================= A. USB Mode Layout =================
            panelModeUSB = new Panel() { Dock = DockStyle.Fill, Visible = true };

            TableLayoutPanel tlpUsbMain = new TableLayoutPanel() { Dock = DockStyle.Fill, RowCount = 2, ColumnCount = 1 };
            tlpUsbMain.RowStyles.Add(new RowStyle(SizeType.Absolute, 380F));
            tlpUsbMain.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));

            // Top: Split Left/Right
            TableLayoutPanel tlpTop = new TableLayoutPanel() { Dock = DockStyle.Fill, RowCount = 1, ColumnCount = 2 };
            tlpTop.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 450F));
            tlpTop.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));

            // Left: Control
            GroupBox grpUsbCtrl = new GroupBox() { Text = "Control Panel", Dock = DockStyle.Fill, Padding = new Padding(20) };
            btnRunUSB = CreateStyledButton("🔴 Start Live Inference", new Point(50, 80), 350, 80, Color.Coral, Color.White);
            btnRunUSB.Font = new Font("Segoe UI", 16, FontStyle.Bold);
            btnRunUSB.Click += BtnRunUSB_Click;

            lblUsbStatus = new Label() { Text = "Status: Idle...", Location = new Point(55, 180), AutoSize = true, ForeColor = Color.Gray, Font = new Font("Segoe UI", 11) };
            grpUsbCtrl.Controls.Add(btnRunUSB);
            grpUsbCtrl.Controls.Add(lblUsbStatus);

            // Right: Results
            GroupBox grpUsbRes = new GroupBox() { Text = "Inference Result", Dock = DockStyle.Fill, Padding = new Padding(10) };
            FlowLayoutPanel flowRes = new FlowLayoutPanel() { Dock = DockStyle.Fill, FlowDirection = FlowDirection.TopDown, AutoScroll = true, WrapContents = false };

            Label l1 = new Label() { Text = "Prediction:", AutoSize = true, Font = new Font("Segoe UI", 14), Margin = new Padding(20, 30, 0, 10) };
            lblPredClass = new Label() { Text = "---", AutoSize = true, Font = new Font("Segoe UI", 36, FontStyle.Bold), ForeColor = Color.RoyalBlue, Margin = new Padding(20, 0, 0, 30) };

            FlowLayoutPanel flowConf = new FlowLayoutPanel() { AutoSize = true, FlowDirection = FlowDirection.LeftToRight };
            Label l2 = new Label() { Text = "Confidence:", AutoSize = true, Font = new Font("Segoe UI", 12), Margin = new Padding(0, 5, 0, 0) };
            lblConfidence = new Label() { Text = "0.0%", AutoSize = true, Font = new Font("Segoe UI", 14, FontStyle.Bold), ForeColor = Color.DarkGreen, Margin = new Padding(10, 0, 0, 0) };
            flowConf.Controls.Add(l2);
            flowConf.Controls.Add(lblConfidence);
            flowConf.Margin = new Padding(20, 0, 0, 10);

            probBar = new ProgressBar() { Size = new Size(500, 30), Value = 0, Margin = new Padding(20, 10, 0, 0) };

            flowRes.Controls.Add(l1);
            flowRes.Controls.Add(lblPredClass);
            flowRes.Controls.Add(flowConf);
            flowRes.Controls.Add(probBar);
            grpUsbRes.Controls.Add(flowRes);

            tlpTop.Controls.Add(grpUsbCtrl, 0, 0);
            tlpTop.Controls.Add(grpUsbRes, 1, 0);

            // Bottom: Log
            GroupBox grpUsbLog = new GroupBox() { Text = "Real-time Logs", Dock = DockStyle.Fill };
            rtbUsbLog = new RichTextBox()
            {
                Dock = DockStyle.Fill,
                BackColor = Color.FromArgb(20, 20, 20),
                ForeColor = Color.LimeGreen,
                Font = new Font("Consolas", 10),
                ReadOnly = true,
                BorderStyle = BorderStyle.None
            };
            grpUsbLog.Controls.Add(rtbUsbLog);

            tlpUsbMain.Controls.Add(tlpTop, 0, 0);
            tlpUsbMain.Controls.Add(grpUsbLog, 0, 1);
            panelModeUSB.Controls.Add(tlpUsbMain);

            // ================= B. Batch Mode Layout =================
            panelModeBatch = new Panel() { Dock = DockStyle.Fill, Visible = false };
            TableLayoutPanel tlpBatch = new TableLayoutPanel() { Dock = DockStyle.Fill, RowCount = 2, ColumnCount = 1 };
            tlpBatch.RowStyles.Add(new RowStyle(SizeType.Absolute, 80F));
            tlpBatch.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));

            btnRunBatch = CreateStyledButton("🚀 Run Batch Generalization", new Point(0, 0), 100, 60, Color.LightSkyBlue, Color.Black);
            btnRunBatch.Dock = DockStyle.Fill;
            btnRunBatch.Margin = new Padding(0, 0, 0, 10);
            btnRunBatch.Click += BtnRunBatch_Click;
            tlpBatch.Controls.Add(btnRunBatch, 0, 0);

            SplitContainer splitBatch = new SplitContainer() { Dock = DockStyle.Fill, Orientation = Orientation.Vertical, SplitterDistance = 500 };
            GroupBox grpLog = new GroupBox() { Text = "Execution Logs", Dock = DockStyle.Fill };
            rtbLog = new RichTextBox() { Dock = DockStyle.Fill, BackColor = Color.FromArgb(30, 30, 30), ForeColor = Color.LimeGreen, Font = new Font("Consolas", 10), ReadOnly = true, BorderStyle = BorderStyle.None };
            grpLog.Controls.Add(rtbLog);

            GroupBox grpImg = new GroupBox() { Text = "Confusion Matrix", Dock = DockStyle.Fill };
            picResult = new PictureBox() { Dock = DockStyle.Fill, SizeMode = PictureBoxSizeMode.Zoom, BackColor = Color.White };
            grpImg.Controls.Add(picResult);

            splitBatch.Panel1.Controls.Add(grpLog);
            splitBatch.Panel2.Controls.Add(grpImg);
            tlpBatch.Controls.Add(splitBatch, 0, 1);
            panelModeBatch.Controls.Add(tlpBatch);

            container.Controls.Add(panelModeUSB);
            container.Controls.Add(panelModeBatch);
        }

        private void ToggleTestMode()
        {
            panelModeUSB.Visible = rbModeUSB.Checked;
            panelModeBatch.Visible = !rbModeUSB.Checked;
        }

        private void BuildTrainTab(TabPage page)
        {
            FlowLayoutPanel flow = new FlowLayoutPanel() { Dock = DockStyle.Fill, AutoScroll = true, Padding = new Padding(20) };

            GroupBox grpCore = CreateGroupBox("Core Parameters", 320);
            numEpochs = AddNumeric(grpCore, "Epochs:", 200, 1, 10000);
            txtLR = AddTextBox(grpCore, "Learning Rate:", "0.001");
            numWarmup = AddNumeric(grpCore, "Warmup Epochs:", 10, 0, 100);

            GroupBox grpOpt = CreateGroupBox("Optimizer & Device", 320);
            cmbOptimizer = AddCombo(grpOpt, "Optimizer:", new string[] { "AdamW", "Adam", "SGD" });
            cmbDevice = AddCombo(grpOpt, "Device:", new string[] { "cuda", "cpu" });
            txtWeightDecay = AddTextBox(grpOpt, "Weight Decay:", "0.001");

            GroupBox grpLoss = CreateGroupBox("Loss Function", 320);
            cmbLossType = AddCombo(grpLoss, "Loss Type:", new string[] { "label_smoothing", "cross_entropy" });

            GroupBox grpLog = CreateGroupBox("Log Path", 350);
            txtLogDir = AddTextBox(grpLog, "Log Dir:", "./tf-logs");
            Button btnLog = CreateStyledButton("...", new Point(300, txtLogDir.Location.Y - 2), 40, 27, Color.LightGray, Color.Black); // Adjusted X for English
            btnLog.Click += (s, e) => {
                using (FolderBrowserDialog fbd = new FolderBrowserDialog()) { if (fbd.ShowDialog() == DialogResult.OK) txtLogDir.Text = fbd.SelectedPath; }
            };
            grpLog.Controls.Add(btnLog);

            flow.Controls.AddRange(new Control[] { grpCore, grpOpt, grpLoss, grpLog });
            page.Controls.Add(flow);
        }

        private void BuildDataTab(TabPage page)
        {
            SplitContainer split = new SplitContainer() { Dock = DockStyle.Fill, SplitterDistance = 450 };
            FlowLayoutPanel flow = new FlowLayoutPanel() { Dock = DockStyle.Fill, AutoScroll = true, Padding = new Padding(20) };

            GroupBox grpBasic = CreateGroupBox("Data Loader", 380);
            numBatchSize = AddNumeric(grpBasic, "Batch Size:", 32, 1, 512);
            numWorkers = AddNumeric(grpBasic, "Num Workers:", 4, 0, 32);
            txtDataDir = AddTextBox(grpBasic, "Data Dir:", "./data");

            GroupBox grpFilter = CreateGroupBox("Whitelist Settings", 380);
            chkUseWhitelist = new CheckBox() { Text = "Enable Whitelist", Location = new Point(20, 40), AutoSize = true, Checked = true };
            grpFilter.Controls.Add(chkUseWhitelist);

            flow.Controls.AddRange(new Control[] { grpBasic, grpFilter });

            GroupBox grpList = new GroupBox() { Text = "Whitelist File Selection", Dock = DockStyle.Fill, Padding = new Padding(10) };
            chkListFiles = new CheckedListBox() { Dock = DockStyle.Fill, CheckOnClick = true, BorderStyle = BorderStyle.None, BackColor = Color.WhiteSmoke };
            grpList.Controls.Add(chkListFiles);

            split.Panel1.Controls.Add(flow);
            split.Panel2.Controls.Add(grpList);
            page.Controls.Add(split);
        }

        private void BuildModelTab(TabPage page)
        {
            TableLayoutPanel tlp = new TableLayoutPanel() { Dock = DockStyle.Fill, RowCount = 1, ColumnCount = 2 };
            tlp.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));
            tlp.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50F));

            // Left: Augment + Architecture
            FlowLayoutPanel flowLeft = new FlowLayoutPanel() { Dock = DockStyle.Fill, AutoScroll = true, Padding = new Padding(20) };

            // --- 1. Augmentation ---
            GroupBox grpAug = CreateGroupBox("Data Augmentation", 400);
            chkSigAug = new CheckBox() { Text = "Enable Signal Augment", Location = new Point(20, 30), AutoSize = true };
            chkImgAug = new CheckBox() { Text = "Enable Image Augment", Location = new Point(20, 60), AutoSize = true };
            grpAug.Controls.AddRange(new Control[] { chkSigAug, chkImgAug });

            // --- 2. Model Architecture ---
            // Increased width to 520 to accommodate longer English labels
            GroupBox grpModel = CreateGroupBox("Model Architecture", 520);
            grpModel.Height = 350;

            // Updated xPos to 260 for better alignment of English text
            numClasses = AddNumeric(grpModel, "Num Classes:", 6, 1, 100, xPos: 260, yStep: 55);
            cmbPseudoMode = AddCombo(grpModel, "Pseudo Image Mode:", new string[] { "0", "1", "2" }, xPos: 260, yStep: 55);
            cmbImgType = AddCombo(grpModel, "Image Model Type:", new string[] { "0", "1", "2" }, xPos: 260, yStep: 55);
            cmbSigType = AddCombo(grpModel, "Signal Model Type:", new string[] { "0", "1", "2" }, xPos: 260, yStep: 55);
            cmbFusionType = AddCombo(grpModel, "Fusion Type:", new string[] { "0", "1", "2" }, xPos: 260, yStep: 55);

            flowLeft.Controls.AddRange(new Control[] { grpAug, grpModel });

            // Right: Ensemble
            GroupBox grpEnsemble = new GroupBox() { Text = "Ensemble Models Settings", Dock = DockStyle.Fill, Margin = new Padding(15) };

            Panel pnlTools = new Panel() { Dock = DockStyle.Top, Height = 45 };
            btnAddModel = CreateStyledButton("➕ Add Model (.pth)", new Point(0, 5), 160, 30, Color.LightSlateGray, Color.White);
            btnRemoveModel = CreateStyledButton("➖ Remove Selected", new Point(170, 5), 150, 30, Color.LightCoral, Color.White);

            btnAddModel.Click += (s, e) => {
                using (OpenFileDialog ofd = new OpenFileDialog())
                {
                    ofd.Filter = "PyTorch Model (*.pth)|*.pth";
                    ofd.Multiselect = true;
                    if (ofd.ShowDialog() == DialogResult.OK)
                    {
                        foreach (var f in ofd.FileNames) if (!lstEnsembleModels.Items.Contains(f)) lstEnsembleModels.Items.Add(f);
                    }
                }
            };
            btnRemoveModel.Click += (s, e) => {
                if (lstEnsembleModels.SelectedIndex != -1) lstEnsembleModels.Items.RemoveAt(lstEnsembleModels.SelectedIndex);
            };

            pnlTools.Controls.Add(btnAddModel);
            pnlTools.Controls.Add(btnRemoveModel);

            lstEnsembleModels = new ListBox() { Dock = DockStyle.Fill, HorizontalScrollbar = true, BorderStyle = BorderStyle.FixedSingle, BackColor = Color.White };
            grpEnsemble.Controls.Add(lstEnsembleModels);
            grpEnsemble.Controls.Add(pnlTools);

            tlp.Controls.Add(flowLeft, 0, 0);
            tlp.Controls.Add(grpEnsemble, 1, 0);
            page.Controls.Add(tlp);
        }
        private void BuildProjectTab(TabPage page)
        {
            FlowLayoutPanel flow = new FlowLayoutPanel() { Dock = DockStyle.Fill, AutoScroll = true, Padding = new Padding(20) };
            GroupBox grpInfo = CreateGroupBox("Project Information", 600);

            txtProjName = AddTextBox(grpInfo, "Project Name:", "");
            txtProjVersion = AddTextBox(grpInfo, "Version:", "");

            Label lblDesc = new Label() { Text = "Description:", Location = new Point(20, 160), AutoSize = true, Font = new Font("Segoe UI", 9) };
            txtProjDesc = new TextBox() { Location = new Point(20, 185), Width = 550, Height = 60, Multiline = true, ScrollBars = ScrollBars.Vertical, BorderStyle = BorderStyle.FixedSingle };

            grpInfo.Controls.AddRange(new Control[] { lblDesc, txtProjDesc });
            flow.Controls.Add(grpInfo);
            page.Controls.Add(flow);
        }

        // ==========================================================
        // 4. Event Handlers
        // ==========================================================

        private void BtnBrowse_Click(object sender, EventArgs e)
        {
            using (FolderBrowserDialog fbd = new FolderBrowserDialog())
            {
                if (fbd.ShowDialog() == DialogResult.OK)
                {
                    string selectedPath = fbd.SelectedPath;
                    if (!File.Exists(Path.Combine(selectedPath, configFileName)))
                    {
                        MessageBox.Show("Error: config.json not found in directory!\nPlease select the project root.", "Config Missing", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return;
                    }
                    projectDir = selectedPath;
                    txtProjectPath.Text = projectDir;
                    LoadConfig();
                }
            }
        }

        private void BtnBrowsePy_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog ofd = new OpenFileDialog())
            {
                ofd.Filter = "Python Executable (python.exe)|python.exe|All Files (*.*)|*.*";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    txtPythonPath.Text = ofd.FileName;
                }
            }
        }

        // --- B. Load Config ---
        private void LoadConfig()
        {
            if (string.IsNullOrEmpty(projectDir)) return;
            try
            {
                string jsonContent = File.ReadAllText(ConfigPath);
                JObject config = JObject.Parse(jsonContent);

                // Python Path
                if (config["project"]["python_path"] != null)
                    txtPythonPath.Text = config["project"]["python_path"].ToString();

                // Train
                numEpochs.Value = (int)config["train"]["epochs"];
                txtLR.Text = config["train"]["learning_rate"].ToString();
                txtWeightDecay.Text = config["train"]["weight_decay"].ToString();
                cmbOptimizer.Text = config["train"]["optimizer"].ToString();
                cmbDevice.Text = config["train"]["device"].ToString();
                cmbLossType.Text = config["train"]["loss_type"].ToString();
                if (config["train"]["log_dir"] != null)
                    txtLogDir.Text = config["train"]["log_dir"].ToString();
                if (config["train"]["warmup_epochs"] != null)
                    numWarmup.Value = (int)config["train"]["warmup_epochs"];

                // Data
                numBatchSize.Value = (int)config["data"]["batch_size"];
                numWorkers.Value = (int)config["data"]["num_workers"];
                txtDataDir.Text = config["data"]["data_dir"].ToString();
                chkUseWhitelist.Checked = (bool)config["data"]["use_whitelist"];
                LoadWhitelist(config);

                // Model Config
                chkSigAug.Checked = (bool)config["augment"]["signal"]["use_augment"];
                chkImgAug.Checked = (bool)config["augment"]["image"]["use_augment"];
                numClasses.Value = (int)config["data"]["num_classes"];

                // Mappings
                if (config["modality"]["pseudo_image_mode"] != null)
                    cmbPseudoMode.Text = config["modality"]["pseudo_image_mode"].ToString();
                else
                    cmbPseudoMode.SelectedIndex = 1;

                if (config["modality"]["image_model"] != null)
                    cmbImgType.Text = config["modality"]["image_model"]["type"].ToString();

                if (config["modality"]["signal_model"] != null)
                    cmbSigType.Text = config["modality"]["signal_model"]["type"].ToString();

                if (config["modality"]["fusion"] != null)
                    cmbFusionType.Text = config["modality"]["fusion"]["type"].ToString();

                // Ensemble
                lstEnsembleModels.Items.Clear();
                if (config["inference"] != null && config["inference"]["ensemble_models"] != null)
                {
                    foreach (var m in config["inference"]["ensemble_models"])
                        lstEnsembleModels.Items.Add(m.ToString());
                }

                // Project
                txtProjName.Text = config["project"]["name"].ToString();
                txtProjVersion.Text = config["project"]["version"].ToString();
                txtProjDesc.Text = config["project"]["description"].ToString();

                MessageBox.Show("✅ Config loaded successfully!", "System Info", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception ex) { MessageBox.Show("Failed to load config: " + ex.Message, "Parse Error", MessageBoxButtons.OK, MessageBoxIcon.Error); }
        }

        private void LoadWhitelist(JObject config)
        {
            chkListFiles.Items.Clear();
            var jsonWhitelist = new HashSet<string>();
            var whitelistArray = config["data"]["whitelist"] as JArray;
            if (whitelistArray != null)
                foreach (var item in whitelistArray) jsonWhitelist.Add(item.ToString());

            try
            {
                string[] scanFiles = Directory.GetFiles(projectDir, "*.npy", SearchOption.AllDirectories);
                if (scanFiles.Length == 0)
                {
                    foreach (var f in jsonWhitelist) chkListFiles.Items.Add(f, true);
                }
                else
                {
                    foreach (var filePath in scanFiles)
                    {
                        string fileName = Path.GetFileName(filePath);
                        chkListFiles.Items.Add(fileName, jsonWhitelist.Contains(fileName));
                    }
                }
            }
            catch { /* Ignore scan errors */ }
        }

        // --- C. Save Config ---
        private void BtnSave_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(projectDir)) { MessageBox.Show("Please select project path first!"); return; }
            try
            {
                string jsonContent = File.ReadAllText(ConfigPath);
                JObject config = JObject.Parse(jsonContent);

                config["project"]["python_path"] = txtPythonPath.Text;

                // Train
                config["train"]["epochs"] = (int)numEpochs.Value;
                if (decimal.TryParse(txtLR.Text, out decimal lr)) config["train"]["learning_rate"] = lr;
                if (decimal.TryParse(txtWeightDecay.Text, out decimal wd)) config["train"]["weight_decay"] = wd;
                config["train"]["log_dir"] = txtLogDir.Text;
                config["train"]["warmup_epochs"] = (int)numWarmup.Value;
                config["train"]["optimizer"] = cmbOptimizer.Text;
                config["train"]["device"] = cmbDevice.Text;
                config["train"]["loss_type"] = cmbLossType.Text;

                // Data
                config["data"]["batch_size"] = (int)numBatchSize.Value;
                config["data"]["num_workers"] = (int)numWorkers.Value;
                config["data"]["use_whitelist"] = chkUseWhitelist.Checked;

                JArray newWhitelist = new JArray();
                foreach (var item in chkListFiles.CheckedItems) newWhitelist.Add(item.ToString());
                config["data"]["whitelist"] = newWhitelist;

                // Model
                config["augment"]["signal"]["use_augment"] = chkSigAug.Checked;
                config["augment"]["image"]["use_augment"] = chkImgAug.Checked;
                config["data"]["num_classes"] = (int)numClasses.Value;

                if (int.TryParse(cmbPseudoMode.Text, out int pm)) config["modality"]["pseudo_image_mode"] = pm;
                if (int.TryParse(cmbImgType.Text, out int it)) config["modality"]["image_model"]["type"] = it;
                if (int.TryParse(cmbSigType.Text, out int st)) config["modality"]["signal_model"]["type"] = st;
                if (int.TryParse(cmbFusionType.Text, out int ft)) config["modality"]["fusion"]["type"] = ft;

                // Ensemble
                if (config["inference"] == null) config["inference"] = new JObject();
                JArray ens = new JArray();
                foreach (var item in lstEnsembleModels.Items) ens.Add(item.ToString());
                config["inference"]["ensemble_models"] = ens;

                // Project
                config["project"]["name"] = txtProjName.Text;
                config["project"]["version"] = txtProjVersion.Text;
                config["project"]["description"] = txtProjDesc.Text;

                File.WriteAllText(ConfigPath, config.ToString());
                MessageBox.Show("✅ Config saved successfully!", "System Info", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception ex) { MessageBox.Show("Save failed: " + ex.Message, "Write Error", MessageBoxButtons.OK, MessageBoxIcon.Error); }
        }

        // ==========================================================
        // 5. Core Testing Logic (USB & Batch)
        // ==========================================================

        private async void BtnRunUSB_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(projectDir)) { MessageBox.Show("Please select project path first!"); return; }

            rtbUsbLog.Clear();
            rtbUsbLog.AppendText(">>> System Ready, Starting Collection Engine...\n");

            lblUsbStatus.Text = "Status: Initializing...";
            lblUsbStatus.ForeColor = Color.Orange;

            btnRunUSB.Enabled = false;
            btnRunUSB.Text = "⏳ Running...";
            btnRunUSB.BackColor = Color.Gray;

            string args = $"{inferenceScript} --mode usb";
            if (chkEnableEnsemble.Checked) args += " --ensemble";

            await Task.Run(() => RunPythonInference(args, isUsb: true));

            btnRunUSB.Enabled = true;
            btnRunUSB.Text = "🔴 Start Live Inference";
            btnRunUSB.BackColor = Color.Coral;
        }

        private async void BtnRunBatch_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(projectDir)) { MessageBox.Show("Please select project path first!"); return; }

            if (picResult.Image != null)
            {
                picResult.Image.Dispose();
                picResult.Image = null;
            }

            btnRunBatch.Enabled = false;
            rtbLog.Clear();
            rtbLog.AppendText(">>> Starting Test...\n");

            string args = $"{inferenceScript} --mode generalization";
            if (chkEnableEnsemble.Checked) args += " --ensemble";

            await Task.Run(() => RunPythonInference(args, isUsb: false));

            btnRunBatch.Enabled = true;

            // Load Image
            if (File.Exists(LogImgFullPath))
            {
                try
                {
                    using (FileStream fs = new FileStream(LogImgFullPath, FileMode.Open, FileAccess.Read))
                    {
                        picResult.Image = Image.FromStream(fs);
                    }
                    rtbLog.AppendText($"\n✅ Confusion Matrix Loaded: {LogImgFullPath}");
                }
                catch (Exception ex)
                {
                    rtbLog.AppendText($"\n❌ Failed to load image: {ex.Message}");
                }
            }
            else
            {
                rtbLog.AppendText($"\n⚠️ Image Not Found: {LogImgFullPath}");
            }
        }

        private void RunPythonInference(string args, bool isUsb)
        {
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = txtPythonPath.Text;
            start.Arguments = args;
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;
            start.RedirectStandardError = true;
            start.CreateNoWindow = true;
            start.WorkingDirectory = projectDir;

            try
            {
                using (Process p = Process.Start(start))
                {
                    p.OutputDataReceived += (s, ev) => {
                        if (ev.Data != null)
                        {
                            if (isUsb && ev.Data.StartsWith("JSON_RESULT:")) ParseUsbResult(ev.Data);
                            else UpdateLog(ev.Data);
                        }
                    };
                    p.ErrorDataReceived += (s, ev) => { if (ev.Data != null) UpdateLog("ERR: " + ev.Data); };

                    p.BeginOutputReadLine();
                    p.BeginErrorReadLine();
                    p.WaitForExit();
                }
            }
            catch (Exception ex)
            {
                UpdateLog($"Startup Failed: {ex.Message}\nPlease check Python path.");
            }
        }

        private void ParseUsbResult(string output)
        {
            try
            {
                string jsonStr = output.Substring("JSON_RESULT:".Length);
                JObject res = JObject.Parse(jsonStr);

                if (res["status"].ToString() == "success")
                {
                    this.Invoke(new Action(() => {
                        lblPredClass.Text = res["prediction"].ToString();
                        double conf = (double)res["confidence"];
                        lblConfidence.Text = $"{conf:P2}";
                        probBar.Value = (int)(conf * 100);
                        lblUsbStatus.Text = $"Finished ({DateTime.Now:HH:mm:ss})";
                        lblUsbStatus.ForeColor = Color.Green;
                    }));
                }
                else
                {
                    UpdateLog($"Inference Error: {res["message"]}");
                }
            }
            catch (Exception ex)
            {
                UpdateLog($"JSON Parse Failed: {ex.Message}");
            }
        }

        private void UpdateLog(string text)
        {
            if (this.InvokeRequired)
            {
                this.Invoke(new Action<string>(UpdateLog), text);
                return;
            }

            if (string.IsNullOrEmpty(text)) return;

            string cleanText = text;
            if (cleanText.StartsWith("LOG:"))
            {
                cleanText = cleanText.Substring(4);
            }

            string timeStr = DateTime.Now.ToString("HH:mm:ss.f");
            string logLine = $"[{timeStr}] {cleanText}{Environment.NewLine}";

            if (rbModeUSB.Checked)
            {
                if (rtbUsbLog != null)
                {
                    rtbUsbLog.AppendText(logLine);
                    rtbUsbLog.ScrollToCaret();
                }
            }
            else
            {
                if (rtbLog != null)
                {
                    rtbLog.AppendText(logLine);
                    rtbLog.ScrollToCaret();
                }
            }
        }

        // ==========================================================
        // 6. UI Helpers (Refined for English Layouts)
        // ==========================================================

        private GroupBox CreateGroupBox(string title, int width)
        {
            return new GroupBox()
            {
                Text = title,
                Width = width,
                Height = 260,
                Margin = new Padding(10),
                Font = new Font("Segoe UI", 9, FontStyle.Bold)
            };
        }

        private TextBox AddTextBox(GroupBox parent, string labelText, string defaultVal, int xPos = 160, int yStep = 45)
        {
            int count = parent.Controls.Count / 2;
            int y = 35 + count * yStep;

            Label lbl = new Label() { Text = labelText, Location = new Point(15, y + 4), AutoSize = true, Font = new Font("Segoe UI", 9) };
            TextBox txt = new TextBox() { Text = defaultVal, Location = new Point(xPos, y), Width = 140, BorderStyle = BorderStyle.FixedSingle };

            parent.Controls.Add(lbl);
            parent.Controls.Add(txt);
            return txt;
        }

        private NumericUpDown AddNumeric(GroupBox parent, string labelText, int defaultVal, int min, int max, int xPos = 160, int yStep = 45)
        {
            int count = parent.Controls.Count / 2;
            int y = 35 + count * yStep;

            Label lbl = new Label() { Text = labelText, Location = new Point(15, y + 4), AutoSize = true, Font = new Font("Segoe UI", 9) };

            NumericUpDown num = new NumericUpDown();
            num.Location = new Point(xPos, y);
            num.Width = 140;
            num.Minimum = min;
            num.Maximum = max;
            num.Value = (defaultVal < min) ? min : (defaultVal > max ? max : defaultVal);

            parent.Controls.Add(lbl);
            parent.Controls.Add(num);
            return num;
        }

        private ComboBox AddCombo(GroupBox parent, string labelText, string[] items, int xPos = 160, int yStep = 45)
        {
            int count = parent.Controls.Count / 2;
            int y = 35 + count * yStep;

            Label lbl = new Label() { Text = labelText, Location = new Point(15, y + 4), AutoSize = true, Font = new Font("Segoe UI", 9) };

            ComboBox cmb = new ComboBox() { Location = new Point(xPos, y), Width = 140, DropDownStyle = ComboBoxStyle.DropDownList };
            cmb.Items.AddRange(items);
            if (items.Length > 0) cmb.SelectedIndex = 0;

            parent.Controls.Add(lbl);
            parent.Controls.Add(cmb);
            return cmb;
        }
    }
}