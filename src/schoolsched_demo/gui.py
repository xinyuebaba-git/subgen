from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import BOTH, END, LEFT, RIGHT, X, Y, messagebox, ttk

from schoolsched_demo.mock_data import (
    DAY_LABELS,
    DAYS,
    GOLDEN_SLOTS,
    ROOM_TYPE_LABELS,
    SLOT_LABELS,
    STRATEGY_LABELS,
    TIME_BANDS,
    Plan,
    ScenarioConfig,
    available_slot_ids,
    apply_manual_adjustment,
    build_demo_dataset,
    build_report_text,
    generate_plan_set,
    preset_to_config,
    room_options,
    slot_to_label,
    teacher_options,
)

PALETTE = {
    "bg": "#F5EEDF",
    "panel": "#FFF8EE",
    "navy": "#163447",
    "teal": "#1E6F66",
    "rust": "#D96C47",
    "gold": "#C99A2E",
    "olive": "#6E7F3F",
    "rose": "#8E5A61",
    "ink": "#243746",
    "muted": "#6E7A83",
    "line": "#D9CBB4",
    "card1": "#18394D",
    "card2": "#197167",
    "card3": "#D17A3F",
    "card4": "#A85B4F",
    "card5": "#6B7D45",
}


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("教培校区智能排课模拟系统")
        self.root.geometry("1500x980")
        self.root.configure(bg=PALETTE["bg"])

        self.dataset = build_demo_dataset()
        self.current_config = preset_to_config(self.dataset.scenario_presets["春季标准周排课"])
        self.plans = generate_plan_set(self.dataset, self.current_config)
        self.plan_by_id = {plan.plan_id: plan for plan in self.plans}
        self.selected_plan_id = self.current_config.focus_plan
        self.last_manual_note = "尚未进行人工调度。"
        self.generation_log: list[str] = []

        self.scenario_var = tk.StringVar(value=self.current_config.scenario_name)
        self.focus_plan_var = tk.StringVar(value=STRATEGY_LABELS[self.current_config.focus_plan])
        self.class_size_var = tk.IntVar(value=self.current_config.class_size_cap)
        self.allow_low_var = tk.BooleanVar(value=self.current_config.allow_low_enrollment)
        self.add_math_teacher_var = tk.BooleanVar(value=self.current_config.add_math_teacher)
        self.add_small_room_var = tk.BooleanVar(value=self.current_config.add_small_room)
        self.add_evening_slot_var = tk.BooleanVar(value=self.current_config.add_evening_slot)
        self.priority_var = tk.StringVar(value=self.current_config.priority_mode)
        self.lock_key_var = tk.BooleanVar(value=self.current_config.lock_key_classes)
        self.revenue_weight_var = tk.DoubleVar(value=45)
        self.satisfaction_weight_var = tk.DoubleVar(value=25)
        self.teacher_weight_var = tk.DoubleVar(value=15)
        self.room_weight_var = tk.DoubleVar(value=15)
        self.plan_selector_var = tk.StringVar(value=STRATEGY_LABELS[self.selected_plan_id])
        self.manual_section_var = tk.StringVar()
        self.manual_slot_var = tk.StringVar()
        self.manual_teacher_var = tk.StringVar()
        self.manual_room_var = tk.StringVar()

        self._build_style()
        self._build_ui()
        self._generate_plans(initial=True)

    def _build_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Avenir Next", 11), foreground=PALETTE["ink"])
        style.configure("TFrame", background=PALETTE["bg"])
        style.configure("TLabelframe", background=PALETTE["panel"], bordercolor=PALETTE["line"])
        style.configure("TLabelframe.Label", background=PALETTE["panel"], foreground=PALETTE["navy"], font=("Avenir Next", 12, "bold"))
        style.configure("TLabel", background=PALETTE["bg"], foreground=PALETTE["ink"])
        style.configure("Panel.TLabel", background=PALETTE["panel"])
        style.configure("TNotebook", background=PALETTE["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 10), background="#E7D8C0", foreground=PALETTE["navy"])
        style.map("TNotebook.Tab", background=[("selected", PALETTE["panel"])], foreground=[("selected", PALETTE["rust"])])
        style.configure("Primary.TButton", background=PALETTE["rust"], foreground="white", borderwidth=0, padding=(12, 8))
        style.map("Primary.TButton", background=[("active", "#B95231")])
        style.configure("Secondary.TButton", background=PALETTE["navy"], foreground="white", borderwidth=0, padding=(10, 7))
        style.map("Secondary.TButton", background=[("active", "#0E2635")])
        style.configure("Treeview", background="white", fieldbackground="white", rowheight=28, bordercolor=PALETTE["line"])
        style.configure("Treeview.Heading", background="#E8DDC8", foreground=PALETTE["navy"], font=("Avenir Next", 11, "bold"))
        style.map("Treeview", background=[("selected", "#F7D7B5")], foreground=[("selected", PALETTE["ink"])])
        style.configure("TCombobox", fieldbackground="white", background="white")

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root, padding=16)
        shell.pack(fill=BOTH, expand=True)

        self._build_header(shell)

        self.notebook = ttk.Notebook(shell)
        self.notebook.pack(fill=BOTH, expand=True, pady=(14, 0))

        self.dashboard_tab = ttk.Frame(self.notebook, padding=12)
        self.config_tab = ttk.Frame(self.notebook, padding=12)
        self.data_tab = ttk.Frame(self.notebook, padding=12)
        self.rules_tab = ttk.Frame(self.notebook, padding=12)
        self.generate_tab = ttk.Frame(self.notebook, padding=12)
        self.compare_tab = ttk.Frame(self.notebook, padding=12)
        self.detail_tab = ttk.Frame(self.notebook, padding=12)
        self.manual_tab = ttk.Frame(self.notebook, padding=12)
        self.conflict_tab = ttk.Frame(self.notebook, padding=12)
        self.export_tab = ttk.Frame(self.notebook, padding=12)

        self.notebook.add(self.dashboard_tab, text="驾驶舱")
        self.notebook.add(self.config_tab, text="场景配置")
        self.notebook.add(self.data_tab, text="基础数据")
        self.notebook.add(self.rules_tab, text="规则权重")
        self.notebook.add(self.generate_tab, text="方案生成")
        self.notebook.add(self.compare_tab, text="方案比较")
        self.notebook.add(self.detail_tab, text="方案详情")
        self.notebook.add(self.manual_tab, text="人工调度")
        self.notebook.add(self.conflict_tab, text="冲突瓶颈")
        self.notebook.add(self.export_tab, text="导出发布")

        self._build_dashboard_tab()
        self._build_config_tab()
        self._build_data_tab()
        self._build_rules_tab()
        self._build_generate_tab()
        self._build_compare_tab()
        self._build_detail_tab()
        self._build_manual_tab()
        self._build_conflict_tab()
        self._build_export_tab()

    def _build_header(self, parent: ttk.Frame) -> None:
        header = tk.Frame(parent, bg=PALETTE["navy"], padx=18, pady=16)
        header.pack(fill=X)

        title_box = tk.Frame(header, bg=PALETTE["navy"])
        title_box.pack(side=LEFT, fill=Y)
        tk.Label(
            title_box,
            text="教培校区智能排课模拟系统",
            bg=PALETTE["navy"],
            fg="white",
            font=("Avenir Next", 22, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="带完整交互的 mock 原型，可直接用于内部评审、流程讨论与方案比较。",
            bg=PALETTE["navy"],
            fg="#D8E4EC",
            font=("Avenir Next", 11),
        ).pack(anchor="w", pady=(4, 0))

        right_box = tk.Frame(header, bg=PALETTE["navy"])
        right_box.pack(side=RIGHT)

        top_row = tk.Frame(right_box, bg=PALETTE["navy"])
        top_row.pack(anchor="e")
        tk.Label(top_row, text="场景预设", bg=PALETTE["navy"], fg="white", font=("Avenir Next", 11, "bold")).pack(side=LEFT, padx=(0, 10))
        self.header_scenario_combo = ttk.Combobox(
            top_row,
            width=18,
            state="readonly",
            values=list(self.dataset.scenario_presets),
            textvariable=self.scenario_var,
        )
        self.header_scenario_combo.pack(side=LEFT, padx=(0, 8))
        ttk.Button(top_row, text="加载场景", style="Secondary.TButton", command=self._load_scenario).pack(side=LEFT, padx=(0, 8))
        ttk.Button(top_row, text="重算方案", style="Primary.TButton", command=self._generate_plans).pack(side=LEFT, padx=(0, 8))
        ttk.Button(top_row, text="重置示例", style="Secondary.TButton", command=self._reset_demo).pack(side=LEFT)

        bottom_row = tk.Frame(right_box, bg=PALETTE["navy"])
        bottom_row.pack(anchor="e", pady=(10, 0))
        self.header_plan_label = tk.Label(
            bottom_row,
            text="当前推荐方案: -",
            bg=PALETTE["navy"],
            fg="#F7D6B2",
            font=("Avenir Next", 12, "bold"),
        )
        self.header_plan_label.pack(side=LEFT)

    def _build_dashboard_tab(self) -> None:
        self.dashboard_cards = tk.Frame(self.dashboard_tab, bg=PALETTE["bg"])
        self.dashboard_cards.pack(fill=X)

        body = ttk.Frame(self.dashboard_tab)
        body.pack(fill=BOTH, expand=True, pady=(12, 0))

        left = ttk.Frame(body)
        left.pack(side=LEFT, fill=BOTH, expand=True)
        right = ttk.Frame(body, width=360)
        right.pack(side=RIGHT, fill=Y, padx=(12, 0))

        chart_box = ttk.LabelFrame(left, text="需求与收入分布", padding=12)
        chart_box.pack(fill=BOTH, expand=True)
        self.demand_canvas = tk.Canvas(chart_box, bg="white", highlightthickness=0, height=360)
        self.demand_canvas.pack(fill=BOTH, expand=True)

        bottom_box = ttk.Frame(left)
        bottom_box.pack(fill=X, pady=(12, 0))

        summary_box = ttk.LabelFrame(bottom_box, text="场景摘要", padding=12)
        summary_box.pack(side=LEFT, fill=BOTH, expand=True)
        self.dashboard_summary_text = tk.Text(summary_box, height=8, wrap="word", bg="white", bd=0, highlightthickness=0, font=("Avenir Next", 11))
        self.dashboard_summary_text.pack(fill=BOTH, expand=True)

        note_box = ttk.LabelFrame(right, text="方案提醒", padding=12)
        note_box.pack(fill=BOTH, expand=True)
        self.alert_listbox = tk.Listbox(
            note_box,
            bg="white",
            bd=0,
            highlightthickness=0,
            font=("Avenir Next", 11),
            activestyle="none",
        )
        self.alert_listbox.pack(fill=BOTH, expand=True)

    def _build_config_tab(self) -> None:
        body = ttk.Frame(self.config_tab)
        body.pack(fill=BOTH, expand=True)

        form = ttk.LabelFrame(body, text="模拟参数", padding=14)
        form.pack(side=LEFT, fill=Y)
        impact = ttk.LabelFrame(body, text="场景影响预览", padding=14)
        impact.pack(side=RIGHT, fill=BOTH, expand=True, padx=(12, 0))

        row = 0
        ttk.Label(form, text="当前场景").grid(row=row, column=0, sticky="w", pady=6)
        ttk.Combobox(form, state="readonly", values=list(self.dataset.scenario_presets), textvariable=self.scenario_var, width=20).grid(row=row, column=1, sticky="ew", pady=6)
        row += 1
        ttk.Label(form, text="默认关注方案").grid(row=row, column=0, sticky="w", pady=6)
        ttk.Combobox(form, state="readonly", values=list(STRATEGY_LABELS.values()), textvariable=self.focus_plan_var, width=20).grid(row=row, column=1, sticky="ew", pady=6)
        row += 1
        ttk.Label(form, text="班额上限").grid(row=row, column=0, sticky="w", pady=6)
        ttk.Spinbox(form, from_=8, to=16, textvariable=self.class_size_var, width=8).grid(row=row, column=1, sticky="w", pady=6)
        row += 1
        ttk.Label(form, text="优先策略").grid(row=row, column=0, sticky="w", pady=6)
        ttk.Combobox(
            form,
            state="readonly",
            textvariable=self.priority_var,
            values=("高需求课程", "高客单课程", "低年级优先", "周末黄金时段优先"),
            width=20,
        ).grid(row=row, column=1, sticky="ew", pady=6)
        row += 1
        ttk.Checkbutton(form, text="允许低于成班下限低开", variable=self.allow_low_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=6)
        row += 1
        ttk.Checkbutton(form, text="新增 1 名数学教师", variable=self.add_math_teacher_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=6)
        row += 1
        ttk.Checkbutton(form, text="新增 1 间小班教室", variable=self.add_small_room_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=6)
        row += 1
        ttk.Checkbutton(form, text="开放周末晚间时段", variable=self.add_evening_slot_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=6)
        row += 1
        ttk.Checkbutton(form, text="锁定重点示范班时段", variable=self.lock_key_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=6)
        row += 1
        action_bar = ttk.Frame(form)
        action_bar.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Button(action_bar, text="载入预设", style="Secondary.TButton", command=self._load_scenario).pack(side=LEFT)
        ttk.Button(action_bar, text="应用并重算", style="Primary.TButton", command=self._generate_plans).pack(side=LEFT, padx=(8, 0))
        form.columnconfigure(1, weight=1)

        self.impact_text = tk.Text(impact, wrap="word", bg="white", bd=0, highlightthickness=0, font=("Avenir Next", 11))
        self.impact_text.pack(fill=BOTH, expand=True)

    def _build_data_tab(self) -> None:
        summary = ttk.LabelFrame(self.data_tab, text="数据概览", padding=12)
        summary.pack(fill=X)
        self.data_summary_label = ttk.Label(summary, text="", style="Panel.TLabel")
        self.data_summary_label.pack(anchor="w")

        notebook = ttk.Notebook(self.data_tab)
        notebook.pack(fill=BOTH, expand=True, pady=(12, 0))

        student_tab = ttk.Frame(notebook, padding=8)
        teacher_tab = ttk.Frame(notebook, padding=8)
        room_tab = ttk.Frame(notebook, padding=8)
        course_tab = ttk.Frame(notebook, padding=8)
        notebook.add(student_tab, text="学生选课")
        notebook.add(teacher_tab, text="教师池")
        notebook.add(room_tab, text="教室池")
        notebook.add(course_tab, text="课程规则")

        self.student_tree = self._create_tree(student_tab, ("id", "name", "grade", "selection_count", "selections", "preference"))
        self.teacher_tree = self._create_tree(teacher_tab, ("id", "name", "skills", "rate", "type", "target"))
        self.room_tree = self._create_tree(room_tab, ("id", "name", "type", "capacity", "target", "availability"))
        self.course_tree = self._create_tree(course_tab, ("code", "name", "room_type", "price", "min_size", "max_size", "demand"))

        self._set_tree_headings(
            self.student_tree,
            {
                "id": ("学生ID", 90),
                "name": ("姓名", 90),
                "grade": ("年级", 70),
                "selection_count": ("选课数", 70),
                "selections": ("课程组合", 360),
                "preference": ("时间偏好", 140),
            },
        )
        self._set_tree_headings(
            self.teacher_tree,
            {
                "id": ("教师ID", 90),
                "name": ("姓名", 90),
                "skills": ("可授课程", 260),
                "rate": ("课酬/小时", 110),
                "type": ("类型", 90),
                "target": ("周目标班数", 110),
            },
        )
        self._set_tree_headings(
            self.room_tree,
            {
                "id": ("教室ID", 90),
                "name": ("教室", 120),
                "type": ("类型", 120),
                "capacity": ("容量", 80),
                "target": ("目标时段", 100),
                "availability": ("开放时段", 420),
            },
        )
        self._set_tree_headings(
            self.course_tree,
            {
                "code": ("课程代码", 90),
                "name": ("课程", 180),
                "room_type": ("要求教室", 180),
                "price": ("单生收入", 110),
                "min_size": ("成班下限", 90),
                "max_size": ("建议上限", 90),
                "demand": ("当前需求", 90),
            },
        )

    def _build_rules_tab(self) -> None:
        body = ttk.Frame(self.rules_tab)
        body.pack(fill=BOTH, expand=True)

        left = ttk.LabelFrame(body, text="硬约束", padding=14)
        left.pack(side=LEFT, fill=BOTH, expand=True)
        right = ttk.LabelFrame(body, text="软目标权重", padding=14)
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(12, 0))

        hard_constraints = [
            "学生同一时段不能出现在两个班级中",
            "教师同一时段不能同时授课两个班级",
            "教室同一时段不能重复占用",
            "课程必须匹配教师资质与教室类型",
            "每周课表固定，不允许周内与周末临时漂移",
            "班级人数不得超出场景设定班额上限",
        ]
        for text in hard_constraints:
            ttk.Checkbutton(left, text=text).pack(anchor="w", pady=6)

        weight_frame = ttk.Frame(right)
        weight_frame.pack(fill=X)
        self._build_weight_slider(weight_frame, "收入权重", self.revenue_weight_var)
        self._build_weight_slider(weight_frame, "满足率权重", self.satisfaction_weight_var)
        self._build_weight_slider(weight_frame, "教师利用权重", self.teacher_weight_var)
        self._build_weight_slider(weight_frame, "教室利用权重", self.room_weight_var)

        info_box = ttk.LabelFrame(right, text="评分说明", padding=12)
        info_box.pack(fill=BOTH, expand=True, pady=(12, 0))
        self.weight_info_text = tk.Text(info_box, wrap="word", bg="white", bd=0, highlightthickness=0, font=("Avenir Next", 11))
        self.weight_info_text.pack(fill=BOTH, expand=True)

    def _build_generate_tab(self) -> None:
        top_bar = ttk.Frame(self.generate_tab)
        top_bar.pack(fill=X)
        ttk.Button(top_bar, text="重新生成方案", style="Primary.TButton", command=self._generate_plans).pack(side=LEFT)
        ttk.Button(top_bar, text="载入选中方案到详情", style="Secondary.TButton", command=lambda: self.notebook.select(self.detail_tab)).pack(side=LEFT, padx=(8, 0))

        body = ttk.Frame(self.generate_tab)
        body.pack(fill=BOTH, expand=True, pady=(12, 0))

        left = ttk.LabelFrame(body, text="候选方案", padding=12)
        left.pack(side=LEFT, fill=BOTH, expand=True)
        right = ttk.LabelFrame(body, text="生成日志", padding=12)
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(12, 0))

        self.plan_tree = self._create_tree(left, ("plan", "revenue", "satisfaction", "classes", "teacher", "room", "score"))
        self._set_tree_headings(
            self.plan_tree,
            {
                "plan": ("方案", 160),
                "revenue": ("总收入", 120),
                "satisfaction": ("满足率", 100),
                "classes": ("开班数", 90),
                "teacher": ("教师利用率", 110),
                "room": ("教室利用率", 110),
                "score": ("综合评分", 100),
            },
        )
        self.plan_tree.bind("<<TreeviewSelect>>", self._on_plan_tree_select)

        self.generation_text = tk.Text(right, wrap="word", bg="white", bd=0, highlightthickness=0, font=("Avenir Next", 11))
        self.generation_text.pack(fill=BOTH, expand=True)

    def _build_compare_tab(self) -> None:
        top = ttk.LabelFrame(self.compare_tab, text="方案推荐", padding=12)
        top.pack(fill=X)
        self.compare_recommend_label = ttk.Label(top, text="", style="Panel.TLabel")
        self.compare_recommend_label.pack(anchor="w")

        self.compare_tree = self._create_tree(self.compare_tab, ("metric", "balance", "income", "open", "resource"))
        self._set_tree_headings(
            self.compare_tree,
            {
                "metric": ("指标", 180),
                "balance": (STRATEGY_LABELS["balance"], 220),
                "income": (STRATEGY_LABELS["income"], 220),
                "open": (STRATEGY_LABELS["open"], 220),
                "resource": (STRATEGY_LABELS["resource"], 220),
            },
        )

        explain_box = ttk.LabelFrame(self.compare_tab, text="差异解释", padding=12)
        explain_box.pack(fill=BOTH, expand=True, pady=(12, 0))
        self.compare_text = tk.Text(explain_box, wrap="word", bg="white", bd=0, highlightthickness=0, font=("Avenir Next", 11))
        self.compare_text.pack(fill=BOTH, expand=True)

    def _build_detail_tab(self) -> None:
        top_bar = ttk.Frame(self.detail_tab)
        top_bar.pack(fill=X)
        ttk.Label(top_bar, text="当前方案").pack(side=LEFT)
        self.detail_plan_combo = ttk.Combobox(top_bar, width=18, state="readonly", textvariable=self.plan_selector_var)
        self.detail_plan_combo.pack(side=LEFT, padx=(8, 0))
        self.detail_plan_combo.bind("<<ComboboxSelected>>", self._on_plan_combo_change)

        detail_notebook = ttk.Notebook(self.detail_tab)
        detail_notebook.pack(fill=BOTH, expand=True, pady=(12, 0))

        schedule_tab = ttk.Frame(detail_notebook, padding=8)
        class_tab = ttk.Frame(detail_notebook, padding=8)
        assignment_tab = ttk.Frame(detail_notebook, padding=8)
        revenue_tab = ttk.Frame(detail_notebook, padding=8)
        detail_notebook.add(schedule_tab, text="周课表")
        detail_notebook.add(class_tab, text="班级明细")
        detail_notebook.add(assignment_tab, text="学生分班")
        detail_notebook.add(revenue_tab, text="收入分析")

        self.schedule_canvas = tk.Canvas(schedule_tab, bg="white", highlightthickness=0)
        self.schedule_canvas.pack(fill=BOTH, expand=True)

        self.class_tree = self._create_tree(class_tab, ("section", "course", "status", "slot", "teacher", "room", "size", "revenue"))
        self._set_tree_headings(
            self.class_tree,
            {
                "section": ("班级", 150),
                "course": ("课程", 180),
                "status": ("状态", 80),
                "slot": ("时段", 150),
                "teacher": ("教师", 100),
                "room": ("教室", 120),
                "size": ("人数/容量", 110),
                "revenue": ("收入", 100),
            },
        )

        self.assignment_tree = self._create_tree(assignment_tab, ("student", "grade", "course", "section", "slot"))
        self._set_tree_headings(
            self.assignment_tree,
            {
                "student": ("学生", 110),
                "grade": ("年级", 80),
                "course": ("课程", 180),
                "section": ("班级", 150),
                "slot": ("时段", 160),
            },
        )

        self.revenue_tree = self._create_tree(revenue_tab, ("course", "demand", "assigned", "sections", "room_type", "revenue"))
        self._set_tree_headings(
            self.revenue_tree,
            {
                "course": ("课程", 190),
                "demand": ("需求", 90),
                "assigned": ("已满足", 90),
                "sections": ("开班", 90),
                "room_type": ("教室要求", 220),
                "revenue": ("收入", 120),
            },
        )

    def _build_manual_tab(self) -> None:
        body = ttk.Frame(self.manual_tab)
        body.pack(fill=BOTH, expand=True)

        left = ttk.LabelFrame(body, text="当前方案班级", padding=12)
        left.pack(side=LEFT, fill=BOTH, expand=True)
        right = ttk.LabelFrame(body, text="调度动作", padding=12)
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(12, 0))

        self.manual_tree = self._create_tree(left, ("section", "slot", "teacher", "room", "size"))
        self._set_tree_headings(
            self.manual_tree,
            {
                "section": ("班级", 160),
                "slot": ("时段", 170),
                "teacher": ("教师", 100),
                "room": ("教室", 120),
                "size": ("人数/容量", 110),
            },
        )
        self.manual_tree.bind("<<TreeviewSelect>>", self._on_manual_tree_select)

        ttk.Label(right, text="选中班级").grid(row=0, column=0, sticky="w", pady=6)
        self.manual_section_combo = ttk.Combobox(right, state="readonly", textvariable=self.manual_section_var, width=24)
        self.manual_section_combo.grid(row=0, column=1, sticky="ew", pady=6)

        ttk.Label(right, text="调整到时段").grid(row=1, column=0, sticky="w", pady=6)
        self.manual_slot_combo = ttk.Combobox(right, state="readonly", textvariable=self.manual_slot_var, width=24)
        self.manual_slot_combo.grid(row=1, column=1, sticky="ew", pady=6)

        ttk.Label(right, text="调整教师").grid(row=2, column=0, sticky="w", pady=6)
        self.manual_teacher_combo = ttk.Combobox(right, state="readonly", textvariable=self.manual_teacher_var, width=24)
        self.manual_teacher_combo.grid(row=2, column=1, sticky="ew", pady=6)

        ttk.Label(right, text="调整教室").grid(row=3, column=0, sticky="w", pady=6)
        self.manual_room_combo = ttk.Combobox(right, state="readonly", textvariable=self.manual_room_var, width=24)
        self.manual_room_combo.grid(row=3, column=1, sticky="ew", pady=6)

        btn_row = ttk.Frame(right)
        btn_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 10))
        ttk.Button(btn_row, text="应用调度", style="Primary.TButton", command=self._apply_manual_adjustment).pack(side=LEFT)
        ttk.Button(btn_row, text="刷新当前数据", style="Secondary.TButton", command=self._refresh_manual_tab).pack(side=LEFT, padx=(8, 0))

        impact = ttk.LabelFrame(right, text="影响分析", padding=12)
        impact.grid(row=5, column=0, columnspan=2, sticky="nsew")
        self.manual_text = tk.Text(impact, wrap="word", bg="white", bd=0, highlightthickness=0, font=("Avenir Next", 11))
        self.manual_text.pack(fill=BOTH, expand=True)
        right.columnconfigure(1, weight=1)
        right.rowconfigure(5, weight=1)

    def _build_conflict_tab(self) -> None:
        top = ttk.Frame(self.conflict_tab)
        top.pack(fill=BOTH, expand=True)

        left = ttk.LabelFrame(top, text="冲突诊断", padding=12)
        left.pack(side=LEFT, fill=BOTH, expand=True)
        right = ttk.LabelFrame(top, text="未满足需求", padding=12)
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(12, 0))

        self.conflict_tree = self._create_tree(left, ("severity", "title", "detail", "suggestion"))
        self._set_tree_headings(
            self.conflict_tree,
            {
                "severity": ("级别", 70),
                "title": ("问题", 160),
                "detail": ("说明", 360),
                "suggestion": ("建议", 300),
            },
        )

        self.blocked_tree = self._create_tree(right, ("student", "course", "reason"))
        self._set_tree_headings(
            self.blocked_tree,
            {
                "student": ("学生", 110),
                "course": ("课程", 180),
                "reason": ("未满足原因", 260),
            },
        )

        rec_box = ttk.LabelFrame(self.conflict_tab, text="资源瓶颈建议", padding=12)
        rec_box.pack(fill=BOTH, expand=True, pady=(12, 0))
        self.conflict_text = tk.Text(rec_box, wrap="word", bg="white", bd=0, highlightthickness=0, font=("Avenir Next", 11))
        self.conflict_text.pack(fill=BOTH, expand=True)

    def _build_export_tab(self) -> None:
        top_bar = ttk.Frame(self.export_tab)
        top_bar.pack(fill=X)
        ttk.Button(top_bar, text="导出当前方案报告", style="Primary.TButton", command=self._export_report).pack(side=LEFT)
        ttk.Button(top_bar, text="打开导出目录", style="Secondary.TButton", command=self._show_export_dir).pack(side=LEFT, padx=(8, 0))

        preview = ttk.LabelFrame(self.export_tab, text="导出预览", padding=12)
        preview.pack(fill=BOTH, expand=True, pady=(12, 0))
        self.export_text = tk.Text(preview, wrap="word", bg="white", bd=0, highlightthickness=0, font=("Menlo", 11))
        self.export_text.pack(fill=BOTH, expand=True)

    def _build_weight_slider(self, parent: ttk.Frame, label: str, variable: tk.DoubleVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=X, pady=6)
        ttk.Label(row, text=label, width=12).pack(side=LEFT)
        ttk.Scale(row, from_=0, to=100, variable=variable).pack(side=LEFT, fill=X, expand=True, padx=(8, 8))
        ttk.Label(row, textvariable=variable, width=5).pack(side=RIGHT)

    def _create_tree(self, parent: ttk.Frame, columns: tuple[str, ...]) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill=BOTH, expand=True)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.pack(side=LEFT, fill=BOTH, expand=True)
        yscroll.pack(side=RIGHT, fill=Y)
        xscroll.pack(side="bottom", fill=X)
        return tree

    def _set_tree_headings(self, tree: ttk.Treeview, mapping: dict[str, tuple[str, int]]) -> None:
        for key, (text, width) in mapping.items():
            tree.heading(key, text=text)
            tree.column(key, width=width, anchor="w")

    def _reset_demo(self) -> None:
        self.scenario_var.set("春季标准周排课")
        self._load_scenario()

    def _load_scenario(self) -> None:
        preset = self.dataset.scenario_presets[self.scenario_var.get()]
        self.current_config = preset_to_config(preset)
        self.focus_plan_var.set(STRATEGY_LABELS[self.current_config.focus_plan])
        self.class_size_var.set(self.current_config.class_size_cap)
        self.allow_low_var.set(self.current_config.allow_low_enrollment)
        self.add_math_teacher_var.set(self.current_config.add_math_teacher)
        self.add_small_room_var.set(self.current_config.add_small_room)
        self.add_evening_slot_var.set(self.current_config.add_evening_slot)
        self.priority_var.set(self.current_config.priority_mode)
        self.lock_key_var.set(self.current_config.lock_key_classes)
        self._generate_plans()

    def _config_from_vars(self) -> ScenarioConfig:
        focus_label = self.focus_plan_var.get()
        focus_plan = next((key for key, label in STRATEGY_LABELS.items() if label == focus_label), "balance")
        return ScenarioConfig(
            scenario_name=self.scenario_var.get(),
            focus_plan=focus_plan,
            class_size_cap=int(self.class_size_var.get()),
            allow_low_enrollment=bool(self.allow_low_var.get()),
            add_math_teacher=bool(self.add_math_teacher_var.get()),
            add_small_room=bool(self.add_small_room_var.get()),
            add_evening_slot=bool(self.add_evening_slot_var.get()),
            priority_mode=self.priority_var.get(),
            lock_key_classes=bool(self.lock_key_var.get()),
        )

    def _generate_plans(self, initial: bool = False) -> None:
        self.current_config = self._config_from_vars()
        self.plans = generate_plan_set(self.dataset, self.current_config)
        self.plan_by_id = {plan.plan_id: plan for plan in self.plans}
        scores = self._compute_scores()
        if self.selected_plan_id not in self.plan_by_id:
            self.selected_plan_id = self.current_config.focus_plan
        if not initial:
            self.selected_plan_id = max(scores, key=scores.get)
        self.plan_selector_var.set(STRATEGY_LABELS[self.selected_plan_id])
        self.generation_log = self._build_generation_log(scores)
        self.last_manual_note = "已按当前参数重新生成 mock 方案，待人工调度。"
        self._refresh_all()

    def _compute_scores(self) -> dict[str, float]:
        revenue_weight = self.revenue_weight_var.get()
        satisfaction_weight = self.satisfaction_weight_var.get()
        teacher_weight = self.teacher_weight_var.get()
        room_weight = self.room_weight_var.get()
        max_revenue = max(plan.metrics.revenue for plan in self.plans) or 1
        scores: dict[str, float] = {}
        for plan in self.plans:
            revenue_score = plan.metrics.revenue / max_revenue
            scores[plan.plan_id] = (
                revenue_score * revenue_weight
                + plan.metrics.satisfaction_rate * satisfaction_weight
                + plan.metrics.teacher_utilization * teacher_weight
                + plan.metrics.room_utilization * room_weight
            )
        return scores

    def _build_generation_log(self, scores: dict[str, float]) -> list[str]:
        best_id = max(scores, key=scores.get)
        best_plan = self.plan_by_id[best_id]
        return [
            f"[1] 载入场景 {self.current_config.scenario_name}，学生 {len(self.dataset.students)} 人，选课请求 {self.dataset.total_requests} 条。",
            f"[2] 班额上限 {self.current_config.class_size_cap}，优先策略 {self.current_config.priority_mode}。",
            f"[3] 资源增量: 数学教师={self.current_config.add_math_teacher}，小班教室={self.current_config.add_small_room}，周末晚间={self.current_config.add_evening_slot}。",
            f"[4] 共生成 {len(self.plans)} 套方案，综合评分最高的是 {best_plan.strategy_label} ({scores[best_id]:.1f})。",
            f"[5] 当前推荐方案收入 {best_plan.metrics.revenue:,} 元，需求满足率 {best_plan.metrics.satisfaction_rate:.1%}。",
        ]

    def _refresh_all(self) -> None:
        self._refresh_dashboard()
        self._refresh_config_tab()
        self._refresh_data_tab()
        self._refresh_rules_tab()
        self._refresh_generate_tab()
        self._refresh_compare_tab()
        self._refresh_detail_tab()
        self._refresh_manual_tab()
        self._refresh_conflict_tab()
        self._refresh_export_tab()
        scores = self._compute_scores()
        best_id = max(scores, key=scores.get)
        self.header_plan_label.configure(text=f"当前推荐方案: {STRATEGY_LABELS[best_id]}")

    def _current_plan(self) -> Plan:
        return self.plan_by_id[self.selected_plan_id]

    def _refresh_dashboard(self) -> None:
        for widget in self.dashboard_cards.winfo_children():
            widget.destroy()
        plan = self._current_plan()
        cards = [
            ("总收入", f"{plan.metrics.revenue:,}", "元 / 模拟月度", PALETTE["card1"]),
            ("需求满足率", f"{plan.metrics.satisfaction_rate:.1%}", "学生选课被满足比例", PALETTE["card2"]),
            ("成功开班", f"{plan.metrics.open_classes}/{plan.metrics.candidate_classes}", "已开班 / 候选班级", PALETTE["card3"]),
            ("教师利用率", f"{plan.metrics.teacher_utilization:.1%}", "重点教师池使用密度", PALETTE["card4"]),
            ("教室利用率", f"{plan.metrics.room_utilization:.1%}", "关键教室时段占用率", PALETTE["card5"]),
        ]
        for title, value, subtitle, color in cards:
            card = tk.Frame(self.dashboard_cards, bg=color, padx=14, pady=12)
            card.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))
            tk.Label(card, text=title, bg=color, fg="#F6F2EC", font=("Avenir Next", 11)).pack(anchor="w")
            tk.Label(card, text=value, bg=color, fg="white", font=("Avenir Next", 24, "bold")).pack(anchor="w", pady=(8, 4))
            tk.Label(card, text=subtitle, bg=color, fg="#E8E2D7", font=("Avenir Next", 10)).pack(anchor="w")
        self._draw_demand_chart(plan)

        summary = [
            f"场景: {self.current_config.scenario_name}",
            f"当前查看方案: {plan.strategy_label}",
            f"优先策略: {self.current_config.priority_mode}",
            f"资源增量: 数学教师={'是' if self.current_config.add_math_teacher else '否'} / "
            f"小班教室={'是' if self.current_config.add_small_room else '否'} / "
            f"周末晚间={'是' if self.current_config.add_evening_slot else '否'}",
            f"班额上限: {self.current_config.class_size_cap}",
            f"低开策略: {'允许' if self.current_config.allow_low_enrollment else '不允许'}",
        ]
        self.dashboard_summary_text.delete("1.0", END)
        self.dashboard_summary_text.insert("1.0", "\n".join(summary))

        self.alert_listbox.delete(0, END)
        for index, item in enumerate(plan.recommendations, start=1):
            self.alert_listbox.insert(END, f"{index}. {item}")
        for conflict in plan.conflicts[:3]:
            self.alert_listbox.insert(END, f"提示: {conflict.title} - {conflict.detail}")

    def _draw_demand_chart(self, plan: Plan) -> None:
        canvas = self.demand_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 900)
        height = max(canvas.winfo_height(), 340)
        canvas.config(scrollregion=(0, 0, width, height))
        margin_left = 120
        margin_top = 28
        bar_gap = 36
        max_value = max(row["demand"] for row in plan.course_breakdown) if plan.course_breakdown else 1
        scale = (width - margin_left - 160) / max_value
        canvas.create_text(20, 10, anchor="nw", text="课程需求 vs 已满足", fill=PALETTE["navy"], font=("Avenir Next", 12, "bold"))
        for idx, row in enumerate(plan.course_breakdown):
            y = margin_top + idx * bar_gap
            demand = int(row["demand"])
            assigned = int(row["assigned"])
            revenue = int(row["revenue"])
            course = self.dataset.courses[row["course_code"]]
            canvas.create_text(16, y + 10, anchor="w", text=row["course_name"], fill=PALETTE["ink"], font=("Avenir Next", 11))
            canvas.create_rectangle(margin_left, y, margin_left + demand * scale, y + 16, fill="#E6DAC5", outline="")
            canvas.create_rectangle(margin_left, y, margin_left + assigned * scale, y + 16, fill=course.color, outline="")
            canvas.create_text(margin_left + demand * scale + 12, y + 8, anchor="w", text=f"{assigned}/{demand}", fill=PALETTE["muted"], font=("Avenir Next", 10))
            canvas.create_text(width - 130, y + 8, anchor="w", text=f"{revenue:,} 元", fill=PALETTE["navy"], font=("Avenir Next", 10, "bold"))

    def _refresh_config_tab(self) -> None:
        self.impact_text.delete("1.0", END)
        notes = [
            f"当前场景将以 {self.focus_plan_var.get()} 作为默认关注方案，但系统仍会同时输出 4 套策略方案供比较。",
            f"班额上限设为 {self.class_size_var.get()}，会直接影响高需求课程的收入和未满足需求量。",
            f"优先策略设为 {self.priority_var.get()}，将改变学生选课请求的分配顺序。",
        ]
        if self.add_math_teacher_var.get():
            notes.append("新增数学教师后，数学与思维训练的容量上浮，教师利用率可能略有下降，但满足率会更稳定。")
        if self.add_small_room_var.get():
            notes.append("新增小班教室将优先承接英语和思维课程的高峰时段班级。")
        if self.add_evening_slot_var.get():
            notes.append("开放周末晚间时段后，编程与思维课程会新增一轮承接窗口。")
        if self.allow_low_var.get():
            notes.append("允许低开将提升开班数，但会牺牲班级平均满班率。")
        if self.lock_key_var.get():
            notes.append("锁定重点示范班会优先保护核心班级时段，适合对外展示与家长沟通。")
        self.impact_text.insert("1.0", "\n\n".join(notes))

    def _refresh_data_tab(self) -> None:
        self.data_summary_label.configure(
            text=(
                f"学生 {len(self.dataset.students)} 人，教师 {len(teacher_options(self.dataset, self.current_config.add_math_teacher))} 名，"
                f"教室 {len(room_options(self.dataset, self.current_config.add_small_room))} 间，"
                f"课程 {len(self.dataset.courses)} 类，当前累计选课请求 {self.dataset.total_requests} 条。"
            )
        )
        self._populate_tree(
            self.student_tree,
            [
                (
                    student.student_id,
                    student.name,
                    student.grade_label,
                    len(student.selections),
                    " / ".join(self.dataset.courses[code].name for code in student.selections),
                    student.preference_tag,
                )
                for student in self.dataset.students
            ],
        )
        self._populate_tree(
            self.teacher_tree,
            [
                (
                    teacher.teacher_id,
                    teacher.name,
                    " / ".join(self.dataset.courses[skill].name for skill in teacher.skills if skill in self.dataset.courses),
                    f"{teacher.hourly_rate} 元",
                    teacher.teacher_type,
                    teacher.target_sections,
                )
                for teacher_id, teacher in self.dataset.teachers.items()
                if self.current_config.add_math_teacher or teacher_id != "T11"
            ],
        )
        self._populate_tree(
            self.room_tree,
            [
                (
                    room.room_id,
                    room.name,
                    ROOM_TYPE_LABELS[room.room_type],
                    room.capacity,
                    room.target_slots,
                    "、".join(slot_to_label(slot) for slot in room.availability[:3]) + ("..." if len(room.availability) > 3 else ""),
                )
                for room_id, room in self.dataset.rooms.items()
                if self.current_config.add_small_room or room_id != "R09"
            ],
        )
        demand = self.dataset.demand_by_course
        self._populate_tree(
            self.course_tree,
            [
                (
                    course.course_code,
                    course.name,
                    " / ".join(ROOM_TYPE_LABELS[item] for item in course.room_types),
                    f"{course.price_per_student} 元",
                    course.min_size,
                    course.recommended_max,
                    demand[course.course_code],
                )
                for course in self.dataset.courses.values()
            ],
        )

    def _refresh_rules_tab(self) -> None:
        self.weight_info_text.delete("1.0", END)
        lines = [
            "当前综合评分用于比较四套 mock 方案，帮助内部讨论更聚焦。",
            "",
            f"收入权重: {self.revenue_weight_var.get():.0f}",
            f"满足率权重: {self.satisfaction_weight_var.get():.0f}",
            f"教师利用权重: {self.teacher_weight_var.get():.0f}",
            f"教室利用权重: {self.room_weight_var.get():.0f}",
            "",
            "建议用法:",
            "1. 如果管理层更关注营收，先拉高收入权重。",
            "2. 如果教务更关心投诉风险，提高满足率权重。",
            "3. 如果校区正在压缩成本，可提高教师和教室利用权重。",
        ]
        self.weight_info_text.insert("1.0", "\n".join(lines))

    def _refresh_generate_tab(self) -> None:
        scores = self._compute_scores()
        rows = []
        for plan in self.plans:
            rows.append(
                (
                    plan.strategy_label,
                    f"{plan.metrics.revenue:,}",
                    f"{plan.metrics.satisfaction_rate:.1%}",
                    f"{plan.metrics.open_classes}/{plan.metrics.candidate_classes}",
                    f"{plan.metrics.teacher_utilization:.1%}",
                    f"{plan.metrics.room_utilization:.1%}",
                    f"{scores[plan.plan_id]:.1f}",
                )
            )
        self._populate_tree(self.plan_tree, rows, ids=[plan.plan_id for plan in self.plans])
        if self.selected_plan_id in self.plan_tree.get_children():
            self.plan_tree.selection_set(self.selected_plan_id)
        self.generation_text.delete("1.0", END)
        self.generation_text.insert("1.0", "\n".join(self.generation_log))

    def _refresh_compare_tab(self) -> None:
        scores = self._compute_scores()
        best_id = max(scores, key=scores.get)
        best_plan = self.plan_by_id[best_id]
        self.compare_recommend_label.configure(
            text=(
                f"当前权重下推荐 {best_plan.strategy_label}，综合评分 {scores[best_id]:.1f}。"
                f" 收入 {best_plan.metrics.revenue:,} 元，满足率 {best_plan.metrics.satisfaction_rate:.1%}。"
            )
        )
        metrics = [
            ("总收入", lambda p: f"{p.metrics.revenue:,} 元"),
            ("需求满足率", lambda p: f"{p.metrics.satisfaction_rate:.1%}"),
            ("开班数", lambda p: f"{p.metrics.open_classes}/{p.metrics.candidate_classes}"),
            ("教师利用率", lambda p: f"{p.metrics.teacher_utilization:.1%}"),
            ("教室利用率", lambda p: f"{p.metrics.room_utilization:.1%}"),
            ("黄金时段利用率", lambda p: f"{p.metrics.golden_utilization:.1%}"),
            ("平均满班率", lambda p: f"{p.metrics.fill_rate:.1%}"),
            ("未满足请求", lambda p: str(p.metrics.unassigned_requests)),
        ]
        rows = []
        for title, formatter in metrics:
            rows.append(
                (
                    title,
                    formatter(self.plan_by_id["balance"]),
                    formatter(self.plan_by_id["income"]),
                    formatter(self.plan_by_id["open"]),
                    formatter(self.plan_by_id["resource"]),
                )
            )
        self._populate_tree(self.compare_tree, rows)

        income_plan = max(self.plans, key=lambda plan: plan.metrics.revenue)
        satisfaction_plan = max(self.plans, key=lambda plan: plan.metrics.satisfaction_rate)
        resource_plan = max(self.plans, key=lambda plan: plan.metrics.teacher_utilization + plan.metrics.room_utilization)
        lines = [
            f"收入最高: {income_plan.strategy_label}，比平衡型多 {income_plan.metrics.revenue - self.plan_by_id['balance'].metrics.revenue:,} 元。",
            f"满足率最高: {satisfaction_plan.strategy_label}，未满足请求 {satisfaction_plan.metrics.unassigned_requests} 条。",
            f"资源最紧凑: {resource_plan.strategy_label}，教师利用率与教室利用率合计最高。",
            "",
            "讨论建议:",
            "1. 如果内部讨论偏经营，先用收入优先型和开班最大化型对比。",
            "2. 如果偏落地执行，重点看平衡型与资源利用优先型。",
            "3. 如果需要演示资源投入价值，可切到“周末高峰扩容模拟”场景再比较一次。",
        ]
        self.compare_text.delete("1.0", END)
        self.compare_text.insert("1.0", "\n".join(lines))

    def _refresh_detail_tab(self) -> None:
        labels = [plan.strategy_label for plan in self.plans]
        self.detail_plan_combo.configure(values=labels)
        self.detail_plan_combo.set(STRATEGY_LABELS[self.selected_plan_id])
        plan = self._current_plan()
        self._draw_schedule(plan)
        self._populate_tree(
            self.class_tree,
            [
                (
                    section.label,
                    self.dataset.courses[section.course_code].name,
                    "已开班" if section.status == "OPENED" else "已取消",
                    slot_to_label(section.slot_id),
                    self.dataset.teachers[section.teacher_id].name,
                    self.dataset.rooms[section.room_id].name,
                    f"{section.assigned_count}/{section.capacity}",
                    f"{section.assigned_count * self.dataset.courses[section.course_code].price_per_student:,}",
                )
                for section in sorted(plan.sections, key=lambda item: item.slot_id)
            ],
        )
        student_grade = {student.student_id: student.grade_label for student in self.dataset.students}
        self._populate_tree(
            self.assignment_tree,
            [
                (
                    assignment.student_name,
                    student_grade[assignment.student_id],
                    assignment.course_name,
                    next(section.label for section in plan.sections if section.section_id == assignment.section_id),
                    slot_to_label(assignment.slot_id),
                )
                for assignment in plan.assignments[:140]
            ],
        )
        self._populate_tree(
            self.revenue_tree,
            [
                (
                    row["course_name"],
                    row["demand"],
                    row["assigned"],
                    row["sections"],
                    row["room_type"],
                    f"{int(row['revenue']):,}",
                )
                for row in plan.course_breakdown
            ],
        )

    def _draw_schedule(self, plan: Plan) -> None:
        canvas = self.schedule_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1160)
        height = max(canvas.winfo_height(), 520)
        canvas.config(scrollregion=(0, 0, width, height))
        left_margin = 90
        top_margin = 36
        col_width = 150
        row_height = 62

        for col, band in enumerate(TIME_BANDS):
            x1 = left_margin + col * col_width
            canvas.create_rectangle(x1, top_margin, x1 + col_width, top_margin + row_height, fill="#F0E4D0", outline=PALETTE["line"])
            canvas.create_text(x1 + col_width / 2, top_margin + row_height / 2, text=band, fill=PALETTE["navy"], font=("Avenir Next", 10, "bold"))
        for row, day in enumerate(DAYS):
            y1 = top_margin + (row + 1) * row_height
            canvas.create_rectangle(0, y1, left_margin, y1 + row_height, fill="#F0E4D0", outline=PALETTE["line"])
            canvas.create_text(left_margin / 2, y1 + row_height / 2, text=DAY_LABELS[day], fill=PALETTE["navy"], font=("Avenir Next", 10, "bold"))
            for col in range(len(TIME_BANDS)):
                x1 = left_margin + col * col_width
                canvas.create_rectangle(x1, y1, x1 + col_width, y1 + row_height, fill="white", outline=PALETTE["line"])

        cell_map: dict[tuple[str, str], list] = {}
        for section in plan.sections:
            if section.status != "OPENED":
                continue
            day, band = section.slot_id.split("@", 1)
            cell_map.setdefault((day, band), []).append(section)

        for (day, band), sections in cell_map.items():
            row = DAYS.index(day)
            col = TIME_BANDS.index(band)
            x1 = left_margin + col * col_width + 4
            y1 = top_margin + (row + 1) * row_height + 4
            box_height = (row_height - 8) / max(len(sections), 1)
            for index, section in enumerate(sections):
                course = self.dataset.courses[section.course_code]
                y_start = y1 + index * box_height
                y_end = y_start + box_height - 4
                canvas.create_rectangle(x1, y_start, x1 + col_width - 8, y_end, fill=course.color, outline="")
                label = f"{section.label}\n{self.dataset.rooms[section.room_id].name} {section.assigned_count}/{section.capacity}"
                canvas.create_text(
                    x1 + 8,
                    y_start + 8,
                    anchor="nw",
                    text=label,
                    fill="white",
                    font=("Avenir Next", 9, "bold"),
                )

    def _refresh_manual_tab(self) -> None:
        plan = self._current_plan()
        labels = [f"{section.label} | {slot_to_label(section.slot_id)}" for section in plan.sections if section.status == "OPENED"]
        self.manual_section_combo.configure(values=labels)
        self.manual_slot_combo.configure(values=[slot_to_label(slot) for slot in available_slot_ids(self.current_config.add_evening_slot)])
        teacher_values = [
            f"{teacher_id} - {self.dataset.teachers[teacher_id].name}"
            for teacher_id in teacher_options(self.dataset, self.current_config.add_math_teacher)
        ]
        room_values = [
            f"{room_id} - {self.dataset.rooms[room_id].name}"
            for room_id in room_options(self.dataset, self.current_config.add_small_room)
        ]
        self.manual_teacher_combo.configure(values=teacher_values)
        self.manual_room_combo.configure(values=room_values)
        self._populate_tree(
            self.manual_tree,
            [
                (
                    section.label,
                    slot_to_label(section.slot_id),
                    self.dataset.teachers[section.teacher_id].name,
                    self.dataset.rooms[section.room_id].name,
                    f"{section.assigned_count}/{section.capacity}",
                )
                for section in plan.sections
                if section.status == "OPENED"
            ],
            ids=[section.section_id for section in plan.sections if section.status == "OPENED"],
        )
        self.manual_text.delete("1.0", END)
        self.manual_text.insert("1.0", self.last_manual_note)

    def _refresh_conflict_tab(self) -> None:
        plan = self._current_plan()
        self._populate_tree(
            self.conflict_tree,
            [(conflict.severity, conflict.title, conflict.detail, conflict.suggestion) for conflict in plan.conflicts],
        )
        self._populate_tree(
            self.blocked_tree,
            [
                (item.student_name, item.course_name, item.reason)
                for item in plan.blocked_requests[:120]
            ],
        )
        self.conflict_text.delete("1.0", END)
        lines = [f"当前方案: {plan.strategy_label}", ""]
        lines.extend(f"- {item}" for item in plan.recommendations)
        if any(section.slot_id in GOLDEN_SLOTS for section in plan.sections if section.status == "OPENED"):
            lines.append("")
            lines.append("黄金时段已被部分高需求课程占用，下一步可以继续比较扩容前后的收入变化。")
        self.conflict_text.insert("1.0", "\n".join(lines))

    def _refresh_export_tab(self) -> None:
        self.export_text.delete("1.0", END)
        self.export_text.insert("1.0", build_report_text(self.dataset, self.current_config, self._current_plan()))

    def _populate_tree(self, tree: ttk.Treeview, rows: list[tuple], ids: list[str] | None = None) -> None:
        tree.delete(*tree.get_children())
        for index, row in enumerate(rows):
            item_id = ids[index] if ids and index < len(ids) else None
            if item_id:
                tree.insert("", END, iid=item_id, values=row)
            else:
                tree.insert("", END, values=row)

    def _on_plan_tree_select(self, _event: object) -> None:
        selected = self.plan_tree.selection()
        if not selected:
            return
        self.selected_plan_id = selected[0]
        self.plan_selector_var.set(STRATEGY_LABELS[self.selected_plan_id])
        self._refresh_dashboard()
        self._refresh_detail_tab()
        self._refresh_manual_tab()
        self._refresh_conflict_tab()
        self._refresh_export_tab()

    def _on_plan_combo_change(self, _event: object) -> None:
        label = self.plan_selector_var.get()
        for plan_id, plan_label in STRATEGY_LABELS.items():
            if plan_label == label:
                self.selected_plan_id = plan_id
                break
        self._refresh_dashboard()
        self._refresh_detail_tab()
        self._refresh_manual_tab()
        self._refresh_conflict_tab()
        self._refresh_export_tab()

    def _on_manual_tree_select(self, _event: object) -> None:
        selected = self.manual_tree.selection()
        if not selected:
            return
        section_id = selected[0]
        section = next(item for item in self._current_plan().sections if item.section_id == section_id)
        self.manual_section_var.set(f"{section.label} | {slot_to_label(section.slot_id)}")
        self.manual_slot_var.set(slot_to_label(section.slot_id))
        self.manual_teacher_var.set(f"{section.teacher_id} - {self.dataset.teachers[section.teacher_id].name}")
        self.manual_room_var.set(f"{section.room_id} - {self.dataset.rooms[section.room_id].name}")

    def _apply_manual_adjustment(self) -> None:
        selected = self.manual_tree.selection()
        if not selected:
            messagebox.showwarning("未选择班级", "请先在左侧表格中选择一个班级。")
            return
        section_id = selected[0]
        slot_label = self.manual_slot_var.get()
        teacher_label = self.manual_teacher_var.get()
        room_label = self.manual_room_var.get()
        if not slot_label or not teacher_label or not room_label:
            messagebox.showwarning("信息不完整", "请选择新的时段、教师和教室。")
            return
        slot_id = next((slot for slot, label in SLOT_LABELS.items() if label == slot_label), "")
        teacher_id = teacher_label.split(" - ", 1)[0]
        room_id = room_label.split(" - ", 1)[0]
        if not slot_id:
            messagebox.showwarning("时段错误", "未找到对应时段，请重新选择。")
            return
        new_plan = apply_manual_adjustment(
            self.dataset,
            self.current_config,
            self._current_plan(),
            section_id=section_id,
            slot_id=slot_id,
            teacher_id=teacher_id,
            room_id=room_id,
        )
        self.plan_by_id[new_plan.plan_id] = new_plan
        self.plans = [self.plan_by_id[plan.plan_id] for plan in self.plans]
        self.last_manual_note = (
            f"已调整 {section_id} -> {slot_label} / {teacher_label} / {room_label}。\n"
            f"新的收入为 {new_plan.metrics.revenue:,} 元，未满足请求 {new_plan.metrics.unassigned_requests} 条。"
        )
        self._refresh_dashboard()
        self._refresh_generate_tab()
        self._refresh_compare_tab()
        self._refresh_detail_tab()
        self._refresh_manual_tab()
        self._refresh_conflict_tab()
        self._refresh_export_tab()

    def _export_report(self) -> None:
        plan = self._current_plan()
        export_dir = Path("dist/school_schedule_demo")
        export_dir.mkdir(parents=True, exist_ok=True)
        file_path = export_dir / f"{plan.plan_id}-report.txt"
        file_path.write_text(build_report_text(self.dataset, self.current_config, plan), encoding="utf-8")
        self.last_manual_note = f"已导出方案报告到 {file_path.resolve()}"
        self._refresh_manual_tab()
        messagebox.showinfo("导出成功", f"已导出到:\n{file_path.resolve()}")

    def _show_export_dir(self) -> None:
        export_dir = Path("dist/school_schedule_demo").resolve()
        messagebox.showinfo("导出目录", str(export_dir))


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
