from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
DAY_LABELS = {
    "Mon": "周一",
    "Tue": "周二",
    "Wed": "周三",
    "Thu": "周四",
    "Fri": "周五",
    "Sat": "周六",
    "Sun": "周日",
}
TIME_BANDS = (
    "09:00-10:30",
    "10:50-12:20",
    "14:00-15:30",
    "15:50-17:20",
    "17:30-19:00",
    "19:20-20:50",
    "19:00-20:30",
)
GOLDEN_SLOTS = {
    "Fri@19:20-20:50",
    "Sat@09:00-10:30",
    "Sat@10:50-12:20",
    "Sat@14:00-15:30",
    "Sun@09:00-10:30",
    "Sun@10:50-12:20",
    "Sun@14:00-15:30",
}
ROOM_TYPE_LABELS = {
    "standard": "标准教室",
    "small": "小班教室",
    "music": "音乐教室",
    "maker": "创客教室",
    "lab": "实验室",
    "multi": "多功能教室",
}
STRATEGY_LABELS = {
    "balance": "平衡型",
    "income": "收入优先型",
    "open": "开班最大化型",
    "resource": "资源利用优先型",
}
STRATEGY_SUMMARIES = {
    "balance": "兼顾收入、开班率与资源均衡，适合标准运营周。",
    "income": "优先保证高客单课程与高满班时段，适合招生转化期。",
    "open": "尽量多开班并覆盖更多学生需求，适合排课冲刺阶段。",
    "resource": "压缩碎片化排班，提升教师与教室使用密度。",
}
SLOT_LABELS = {
    f"{day}@{band}": f"{DAY_LABELS[day]} {band}"
    for day in DAYS
    for band in TIME_BANDS
}


@dataclass(frozen=True)
class Student:
    student_id: str
    name: str
    grade_level: int
    selections: tuple[str, ...]
    preference_tag: str

    @property
    def grade_label(self) -> str:
        return f"{self.grade_level}年级"


@dataclass(frozen=True)
class Teacher:
    teacher_id: str
    name: str
    skills: tuple[str, ...]
    hourly_rate: int
    teacher_type: str
    target_sections: int
    availability: tuple[str, ...]


@dataclass(frozen=True)
class Room:
    room_id: str
    name: str
    room_type: str
    capacity: int
    target_slots: int
    availability: tuple[str, ...]


@dataclass(frozen=True)
class Course:
    course_code: str
    name: str
    room_types: tuple[str, ...]
    teacher_skill: str
    price_per_student: int
    min_size: int
    recommended_max: int
    color: str


@dataclass(frozen=True)
class SectionTemplate:
    section_id: str
    course_code: str
    label: str
    teacher_id: str
    room_id: str
    slot_id: str
    base_capacity: int
    tags: tuple[str, ...] = ()


@dataclass
class SectionSpec:
    section_id: str
    course_code: str
    label: str
    teacher_id: str
    room_id: str
    slot_id: str
    capacity: int


@dataclass
class ScheduledSection:
    section_id: str
    course_code: str
    label: str
    teacher_id: str
    room_id: str
    slot_id: str
    capacity: int
    assigned_student_ids: list[str] = field(default_factory=list)
    status: str = "OPENED"
    note: str = ""

    @property
    def assigned_count(self) -> int:
        return len(self.assigned_student_ids)


@dataclass(frozen=True)
class EnrollmentAssignment:
    student_id: str
    student_name: str
    course_code: str
    course_name: str
    section_id: str
    slot_id: str


@dataclass(frozen=True)
class BlockedRequest:
    student_id: str
    student_name: str
    course_code: str
    course_name: str
    reason: str


@dataclass(frozen=True)
class Conflict:
    severity: str
    title: str
    detail: str
    suggestion: str


@dataclass
class PlanMetrics:
    revenue: int
    assigned_requests: int
    total_requests: int
    open_classes: int
    candidate_classes: int
    teacher_utilization: float
    room_utilization: float
    golden_utilization: float
    fill_rate: float
    unassigned_requests: int

    @property
    def satisfaction_rate(self) -> float:
        if self.total_requests <= 0:
            return 0.0
        return self.assigned_requests / self.total_requests

    @property
    def open_rate(self) -> float:
        if self.candidate_classes <= 0:
            return 0.0
        return self.open_classes / self.candidate_classes


@dataclass
class Plan:
    plan_id: str
    strategy_key: str
    strategy_label: str
    summary: str
    metrics: PlanMetrics
    sections: list[ScheduledSection]
    assignments: list[EnrollmentAssignment]
    blocked_requests: list[BlockedRequest]
    conflicts: list[Conflict]
    course_breakdown: list[dict[str, object]]
    recommendations: list[str]
    generation_notes: list[str]
    section_specs: list[SectionSpec]


@dataclass(frozen=True)
class ScenarioPreset:
    name: str
    focus_plan: str
    class_size_cap: int
    allow_low_enrollment: bool
    add_math_teacher: bool
    add_small_room: bool
    add_evening_slot: bool
    priority_mode: str
    lock_key_classes: bool


@dataclass
class ScenarioConfig:
    scenario_name: str
    focus_plan: str
    class_size_cap: int
    allow_low_enrollment: bool
    add_math_teacher: bool
    add_small_room: bool
    add_evening_slot: bool
    priority_mode: str
    lock_key_classes: bool


@dataclass
class DemoDataset:
    students: list[Student]
    teachers: dict[str, Teacher]
    rooms: dict[str, Room]
    courses: dict[str, Course]
    section_templates: dict[str, SectionTemplate]
    section_groups: dict[str, list[str]]
    scenario_presets: dict[str, ScenarioPreset]

    @property
    def total_requests(self) -> int:
        return sum(len(student.selections) for student in self.students)

    @property
    def demand_by_course(self) -> Counter[str]:
        demand: Counter[str] = Counter()
        for student in self.students:
            demand.update(student.selections)
        return demand


def build_demo_dataset() -> DemoDataset:
    slots = _base_slots()
    teachers = _build_teachers(slots)
    rooms = _build_rooms(slots)
    courses = _build_courses()
    students = _build_students()
    section_templates = _build_section_templates()
    section_groups = {
        "balance": [
            "MATH-A",
            "MATH-B",
            "MATH-C",
            "ENG-A",
            "ENG-B",
            "ENG-C",
            "THINK-A",
            "PROG-A",
            "PROG-B",
            "ROBOT-A",
            "ROBOT-B",
            "PHYS-A",
            "PIANO-A",
            "CALLI-A",
            "CALLI-B",
        ],
        "income": [
            "MATH-A",
            "MATH-B",
            "MATH-C",
            "ENG-A",
            "ENG-B",
            "ENG-C",
            "PROG-A",
            "PROG-B",
            "ROBOT-A",
            "ROBOT-B",
            "PHYS-A",
            "PHYS-B",
            "PIANO-A",
            "PIANO-B",
            "CALLI-A",
        ],
        "open": [
            "MATH-A",
            "MATH-B",
            "MATH-C",
            "ENG-A",
            "ENG-B",
            "ENG-C",
            "THINK-A",
            "THINK-B",
            "PROG-A",
            "PROG-B",
            "ROBOT-A",
            "ROBOT-B",
            "PHYS-A",
            "PHYS-B",
            "PIANO-A",
            "PIANO-B",
            "CALLI-A",
            "CALLI-B",
        ],
        "resource": [
            "MATH-A",
            "MATH-B",
            "ENG-A",
            "ENG-B",
            "ENG-C",
            "THINK-A",
            "PROG-A",
            "PROG-B",
            "ROBOT-A",
            "PHYS-A",
            "PIANO-A",
            "CALLI-A",
        ],
    }
    scenario_presets = {
        "春季标准周排课": ScenarioPreset(
            name="春季标准周排课",
            focus_plan="balance",
            class_size_cap=12,
            allow_low_enrollment=False,
            add_math_teacher=False,
            add_small_room=False,
            add_evening_slot=False,
            priority_mode="高需求课程",
            lock_key_classes=True,
        ),
        "周末高峰扩容模拟": ScenarioPreset(
            name="周末高峰扩容模拟",
            focus_plan="open",
            class_size_cap=12,
            allow_low_enrollment=False,
            add_math_teacher=False,
            add_small_room=True,
            add_evening_slot=True,
            priority_mode="周末黄金时段优先",
            lock_key_classes=False,
        ),
        "招生冲刺收入优先": ScenarioPreset(
            name="招生冲刺收入优先",
            focus_plan="income",
            class_size_cap=13,
            allow_low_enrollment=False,
            add_math_teacher=True,
            add_small_room=False,
            add_evening_slot=True,
            priority_mode="高客单课程",
            lock_key_classes=True,
        ),
        "保守资源运营": ScenarioPreset(
            name="保守资源运营",
            focus_plan="resource",
            class_size_cap=13,
            allow_low_enrollment=False,
            add_math_teacher=False,
            add_small_room=False,
            add_evening_slot=False,
            priority_mode="高需求课程",
            lock_key_classes=True,
        ),
    }
    return DemoDataset(
        students=students,
        teachers=teachers,
        rooms=rooms,
        courses=courses,
        section_templates=section_templates,
        section_groups=section_groups,
        scenario_presets=scenario_presets,
    )


def preset_to_config(preset: ScenarioPreset) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_name=preset.name,
        focus_plan=preset.focus_plan,
        class_size_cap=preset.class_size_cap,
        allow_low_enrollment=preset.allow_low_enrollment,
        add_math_teacher=preset.add_math_teacher,
        add_small_room=preset.add_small_room,
        add_evening_slot=preset.add_evening_slot,
        priority_mode=preset.priority_mode,
        lock_key_classes=preset.lock_key_classes,
    )


def generate_plan_set(dataset: DemoDataset, config: ScenarioConfig) -> list[Plan]:
    plans: list[Plan] = []
    for strategy_key in ("balance", "income", "open", "resource"):
        specs = build_specs_for_strategy(dataset, config, strategy_key)
        plan = simulate_plan(dataset, config, strategy_key, specs)
        plans.append(plan)
    return plans


def build_specs_for_strategy(
    dataset: DemoDataset,
    config: ScenarioConfig,
    strategy_key: str,
) -> list[SectionSpec]:
    active_teachers = set(dataset.teachers)
    active_rooms = set(dataset.rooms)
    if not config.add_math_teacher:
        active_teachers.discard("T11")
    if not config.add_small_room:
        active_rooms.discard("R09")

    specs: list[SectionSpec] = []
    for section_id in dataset.section_groups[strategy_key]:
        template = dataset.section_templates[section_id]
        if "evening_slot" in template.tags and not config.add_evening_slot:
            continue
        if "extra_math_teacher" in template.tags and not config.add_math_teacher:
            continue
        if "extra_small_room" in template.tags and not config.add_small_room:
            continue
        if template.teacher_id not in active_teachers or template.room_id not in active_rooms:
            continue
        specs.append(_build_spec(dataset, config, strategy_key, template))

    if config.add_math_teacher and strategy_key in {"balance", "income", "open"}:
        specs.append(_build_spec(dataset, config, strategy_key, dataset.section_templates["MATH-D"]))

    if config.add_small_room and strategy_key in {"balance", "open"}:
        specs.append(_build_spec(dataset, config, strategy_key, dataset.section_templates["ENG-D"]))

    if config.add_evening_slot and strategy_key in {"balance", "income", "open"}:
        specs.append(_build_spec(dataset, config, strategy_key, dataset.section_templates["THINK-C"]))
        specs.append(_build_spec(dataset, config, strategy_key, dataset.section_templates["PROG-C"]))

    if not config.allow_low_enrollment and strategy_key == "open":
        for spec in specs:
            if spec.course_code in {"PIANO", "CALLI"}:
                spec.capacity = max(4, spec.capacity - 1)

    if config.lock_key_classes:
        for spec in specs:
            if spec.section_id in {"MATH-A", "ENG-A", "PROG-A"}:
                spec.capacity = min(spec.capacity + 1, dataset.rooms[spec.room_id].capacity)

    return specs


def simulate_plan(
    dataset: DemoDataset,
    config: ScenarioConfig,
    strategy_key: str,
    section_specs: list[SectionSpec],
) -> Plan:
    working_specs = [
        SectionSpec(
            section_id=spec.section_id,
            course_code=spec.course_code,
            label=spec.label,
            teacher_id=spec.teacher_id,
            room_id=spec.room_id,
            slot_id=spec.slot_id,
            capacity=spec.capacity,
        )
        for spec in section_specs
    ]
    sections, assignments, blocked = _assign_requests(dataset, config, working_specs, strategy_key)
    if not config.allow_low_enrollment:
        sections, assignments, blocked = _cancel_under_enrolled_sections(
            dataset,
            config,
            sections,
            strategy_key,
        )

    metrics = _build_metrics(dataset, sections, assignments, blocked, config)
    breakdown = _build_course_breakdown(dataset, sections, blocked)
    conflicts = _build_conflicts(dataset, config, sections, blocked)
    notes = _build_generation_notes(dataset, config, strategy_key, sections, blocked)
    recommendations = _build_recommendations(dataset, config, sections, blocked)

    return Plan(
        plan_id=strategy_key,
        strategy_key=strategy_key,
        strategy_label=STRATEGY_LABELS[strategy_key],
        summary=STRATEGY_SUMMARIES[strategy_key],
        metrics=metrics,
        sections=sections,
        assignments=assignments,
        blocked_requests=blocked,
        conflicts=conflicts,
        course_breakdown=breakdown,
        recommendations=recommendations,
        generation_notes=notes,
        section_specs=working_specs,
    )


def apply_manual_adjustment(
    dataset: DemoDataset,
    config: ScenarioConfig,
    plan: Plan,
    section_id: str,
    slot_id: str,
    teacher_id: str,
    room_id: str,
) -> Plan:
    updated_specs: list[SectionSpec] = []
    for spec in plan.section_specs:
        if spec.section_id == section_id:
            updated_specs.append(
                SectionSpec(
                    section_id=spec.section_id,
                    course_code=spec.course_code,
                    label=spec.label,
                    teacher_id=teacher_id,
                    room_id=room_id,
                    slot_id=slot_id,
                    capacity=min(spec.capacity, dataset.rooms[room_id].capacity),
                )
            )
        else:
            updated_specs.append(
                SectionSpec(
                    section_id=spec.section_id,
                    course_code=spec.course_code,
                    label=spec.label,
                    teacher_id=spec.teacher_id,
                    room_id=spec.room_id,
                    slot_id=spec.slot_id,
                    capacity=spec.capacity,
                )
            )
    return simulate_plan(dataset, config, plan.strategy_key, updated_specs)


def build_report_text(dataset: DemoDataset, config: ScenarioConfig, plan: Plan) -> str:
    lines = [
        "智能排课模拟报告",
        "=" * 40,
        f"场景: {config.scenario_name}",
        f"方案: {plan.strategy_label}",
        f"策略摘要: {plan.summary}",
        "",
        "核心指标",
        "-" * 40,
        f"总收入(模拟月度): {plan.metrics.revenue:,} 元",
        f"成功满足选课请求: {plan.metrics.assigned_requests}/{plan.metrics.total_requests}",
        f"需求满足率: {plan.metrics.satisfaction_rate:.1%}",
        f"成功开班数: {plan.metrics.open_classes}/{plan.metrics.candidate_classes}",
        f"教师利用率: {plan.metrics.teacher_utilization:.1%}",
        f"教室利用率: {plan.metrics.room_utilization:.1%}",
        f"黄金时段利用率: {plan.metrics.golden_utilization:.1%}",
        "",
        "课程收入",
        "-" * 40,
    ]
    for row in plan.course_breakdown:
        lines.append(
            f"{row['course_name']}: 收入 {int(row['revenue']):,} 元, "
            f"已分配 {int(row['assigned'])}/{int(row['demand'])}, 开班 {int(row['sections'])}"
        )
    lines.extend(["", "关键冲突", "-" * 40])
    if plan.conflicts:
        for conflict in plan.conflicts:
            lines.append(
                f"[{conflict.severity}] {conflict.title}: {conflict.detail} | 建议: {conflict.suggestion}"
            )
    else:
        lines.append("无明显冲突。")
    lines.extend(["", "建议动作", "-" * 40])
    for recommendation in plan.recommendations:
        lines.append(f"- {recommendation}")
    lines.extend(["", "样例班级明细", "-" * 40])
    for section in plan.sections[:12]:
        course = dataset.courses[section.course_code]
        teacher = dataset.teachers[section.teacher_id]
        room = dataset.rooms[section.room_id]
        lines.append(
            f"{section.label} | {course.name} | {slot_to_label(section.slot_id)} | "
            f"{teacher.name} | {room.name} | {section.assigned_count}/{section.capacity}"
        )
    return "\n".join(lines)


def slot_to_label(slot_id: str) -> str:
    return SLOT_LABELS.get(slot_id, slot_id)


def preference_match_score(preference_tag: str, slot_id: str) -> int:
    day, band = slot_id.split("@", 1)
    is_weekend = day in {"Sat", "Sun"}
    is_morning = band in {"09:00-10:30", "10:50-12:20"}
    is_afternoon = band in {"14:00-15:30", "15:50-17:20"}
    is_evening = band in {"17:30-19:00", "19:20-20:50", "19:00-20:30"}

    rules = {
        "周内晚间": (not is_weekend and is_evening, 5),
        "周末上午": (is_weekend and is_morning, 5),
        "周末下午": (is_weekend and is_afternoon, 5),
        "周末全天": (is_weekend, 4),
        "周三/周六": (day in {"Wed", "Sat"}, 4),
        "周二/周四": (day in {"Tue", "Thu"}, 4),
        "周五/周日": (day in {"Fri", "Sun"}, 4),
        "周末优先": (is_weekend, 5),
        "工作日优先": (not is_weekend, 4),
    }
    match = rules.get(preference_tag)
    if match and match[0]:
        return match[1]
    if is_weekend and "周末" in preference_tag:
        return 3
    if not is_weekend and "周内" in preference_tag:
        return 3
    return 1


def available_slot_ids(include_evening: bool) -> list[str]:
    slot_ids = list(_base_slots())
    if include_evening:
        slot_ids.extend(["Sat@19:00-20:30", "Sun@19:00-20:30"])
    return slot_ids


def _base_slots() -> list[str]:
    return [
        "Mon@17:30-19:00",
        "Mon@19:20-20:50",
        "Tue@17:30-19:00",
        "Tue@19:20-20:50",
        "Wed@17:30-19:00",
        "Wed@19:20-20:50",
        "Thu@17:30-19:00",
        "Thu@19:20-20:50",
        "Fri@17:30-19:00",
        "Fri@19:20-20:50",
        "Sat@09:00-10:30",
        "Sat@10:50-12:20",
        "Sat@14:00-15:30",
        "Sat@15:50-17:20",
        "Sun@09:00-10:30",
        "Sun@10:50-12:20",
        "Sun@14:00-15:30",
        "Sun@15:50-17:20",
    ]


def _build_teachers(slots: Iterable[str]) -> dict[str, Teacher]:
    slot_set = set(slots) | {"Sat@19:00-20:30", "Sun@19:00-20:30"}
    teachers = [
        Teacher(
            teacher_id="T01",
            name="林晨",
            skills=("MATH", "THINK"),
            hourly_rate=260,
            teacher_type="全职",
            target_sections=3,
            availability=tuple(
                slot
                for slot in slot_set
                if slot.startswith(("Mon@", "Tue@", "Wed@", "Fri@", "Sat@", "Sun@"))
                and not slot.startswith("Fri@19:20")
            ),
        ),
        Teacher(
            teacher_id="T02",
            name="何敏",
            skills=("ENG",),
            hourly_rate=230,
            teacher_type="全职",
            target_sections=3,
            availability=tuple(
                slot
                for slot in slot_set
                if slot.startswith(("Tue@", "Thu@", "Sat@", "Sun@"))
            ),
        ),
        Teacher(
            teacher_id="T03",
            name="周睿",
            skills=("PROG", "ROBOT"),
            hourly_rate=300,
            teacher_type="全职",
            target_sections=3,
            availability=tuple(
                slot
                for slot in slot_set
                if slot.startswith(("Fri@", "Sat@", "Sun@"))
            ),
        ),
        Teacher(
            teacher_id="T04",
            name="陈婷",
            skills=("PIANO",),
            hourly_rate=360,
            teacher_type="兼职",
            target_sections=2,
            availability=(
                "Sat@09:00-10:30",
                "Sun@09:00-10:30",
                "Sun@10:50-12:20",
            ),
        ),
        Teacher(
            teacher_id="T05",
            name="王岩",
            skills=("PHYS", "ROBOT"),
            hourly_rate=280,
            teacher_type="全职",
            target_sections=3,
            availability=tuple(
                slot
                for slot in slot_set
                if slot.startswith(("Thu@", "Sat@", "Sun@"))
            ),
        ),
        Teacher(
            teacher_id="T06",
            name="赵宁",
            skills=("CALLI",),
            hourly_rate=210,
            teacher_type="兼职",
            target_sections=2,
            availability=(
                "Wed@19:20-20:50",
                "Fri@19:20-20:50",
                "Sun@10:50-12:20",
                "Sun@14:00-15:30",
            ),
        ),
        Teacher(
            teacher_id="T07",
            name="胡楠",
            skills=("MATH", "PHYS"),
            hourly_rate=320,
            teacher_type="兼职",
            target_sections=2,
            availability=(
                "Thu@19:20-20:50",
                "Sat@09:00-10:30",
                "Sat@14:00-15:30",
                "Sun@09:00-10:30",
                "Sun@10:50-12:20",
            ),
        ),
        Teacher(
            teacher_id="T08",
            name="孙岚",
            skills=("ENG",),
            hourly_rate=220,
            teacher_type="兼职",
            target_sections=2,
            availability=(
                "Sat@10:50-12:20",
                "Sun@14:00-15:30",
                "Fri@19:20-20:50",
            ),
        ),
        Teacher(
            teacher_id="T09",
            name="李可",
            skills=("PROG",),
            hourly_rate=290,
            teacher_type="兼职",
            target_sections=2,
            availability=(
                "Sat@14:00-15:30",
                "Sun@19:00-20:30",
                "Sun@15:50-17:20",
            ),
        ),
        Teacher(
            teacher_id="T10",
            name="方洁",
            skills=("THINK",),
            hourly_rate=200,
            teacher_type="全职",
            target_sections=2,
            availability=(
                "Wed@17:30-19:00",
                "Sat@14:00-15:30",
                "Sat@19:00-20:30",
                "Sun@10:50-12:20",
            ),
        ),
        Teacher(
            teacher_id="T11",
            name="蒋博",
            skills=("MATH", "THINK"),
            hourly_rate=275,
            teacher_type="兼职",
            target_sections=2,
            availability=(
                "Sat@10:50-12:20",
                "Sun@10:50-12:20",
                "Sun@14:00-15:30",
            ),
        ),
    ]
    return {teacher.teacher_id: teacher for teacher in teachers}


def _build_rooms(slots: Iterable[str]) -> dict[str, Room]:
    slot_tuple = tuple(slots)
    rooms = [
        Room("R01", "A101", "standard", 16, 4, slot_tuple),
        Room("R02", "A102", "standard", 16, 4, slot_tuple),
        Room("R03", "B201", "small", 12, 3, slot_tuple + ("Sat@19:00-20:30",)),
        Room("R04", "B202", "small", 12, 3, slot_tuple),
        Room("R05", "C301", "music", 8, 2, ("Sat@09:00-10:30", "Sun@09:00-10:30", "Sun@10:50-12:20")),
        Room("R06", "C302", "maker", 14, 3, slot_tuple + ("Sun@19:00-20:30",)),
        Room("R07", "D401", "lab", 12, 2, slot_tuple),
        Room("R08", "共享活动室", "multi", 18, 2, slot_tuple),
        Room("R09", "B203", "small", 10, 2, slot_tuple),
    ]
    return {room.room_id: room for room in rooms}


def _build_courses() -> dict[str, Course]:
    courses = [
        Course("MATH", "小学数学培优", ("standard", "small"), "MATH", 1680, 8, 12, "#D17A3F"),
        Course("ENG", "英语阅读写作", ("standard", "small"), "ENG", 1580, 8, 12, "#0D7C66"),
        Course("THINK", "思维训练", ("small",), "THINK", 1480, 6, 10, "#C89B2C"),
        Course("PROG", "少儿编程 Python", ("maker",), "PROG", 1880, 6, 10, "#1F5F8B"),
        Course("ROBOT", "机器人创客", ("maker",), "ROBOT", 1980, 6, 10, "#944F48"),
        Course("PHYS", "初中物理实验", ("lab",), "PHYS", 1880, 6, 10, "#5E7460"),
        Course("PIANO", "钢琴启蒙", ("music",), "PIANO", 2280, 4, 6, "#9C6C2F"),
        Course("CALLI", "书法基础", ("standard", "small", "multi"), "CALLI", 1280, 6, 10, "#7C5258"),
    ]
    return {course.course_code: course for course in courses}


def _build_section_templates() -> dict[str, SectionTemplate]:
    templates = [
        SectionTemplate("MATH-A", "MATH", "数学培优 A", "T01", "R01", "Mon@17:30-19:00", 12),
        SectionTemplate("MATH-B", "MATH", "数学培优 B", "T07", "R01", "Sat@09:00-10:30", 12),
        SectionTemplate("MATH-C", "MATH", "数学培优 C", "T01", "R03", "Sun@09:00-10:30", 10),
        SectionTemplate("MATH-D", "MATH", "数学培优 D", "T11", "R02", "Sat@10:50-12:20", 10, ("extra_math_teacher",)),
        SectionTemplate("ENG-A", "ENG", "英语阅读 A", "T02", "R02", "Tue@17:30-19:00", 12),
        SectionTemplate("ENG-B", "ENG", "英语阅读 B", "T08", "R02", "Sat@10:50-12:20", 12),
        SectionTemplate("ENG-C", "ENG", "英语阅读 C", "T02", "R04", "Sun@14:00-15:30", 10),
        SectionTemplate("ENG-D", "ENG", "英语阅读 D", "T08", "R09", "Fri@19:20-20:50", 10, ("extra_small_room",)),
        SectionTemplate("THINK-A", "THINK", "思维训练 A", "T10", "R03", "Wed@17:30-19:00", 10),
        SectionTemplate("THINK-B", "THINK", "思维训练 B", "T10", "R03", "Sat@14:00-15:30", 8),
        SectionTemplate("THINK-C", "THINK", "思维训练 C", "T10", "R03", "Sat@19:00-20:30", 8, ("evening_slot",)),
        SectionTemplate("PROG-A", "PROG", "编程 Python A", "T03", "R06", "Fri@17:30-19:00", 10),
        SectionTemplate("PROG-B", "PROG", "编程 Python B", "T09", "R06", "Sat@14:00-15:30", 10),
        SectionTemplate("PROG-C", "PROG", "编程 Python C", "T09", "R06", "Sun@19:00-20:30", 10, ("evening_slot",)),
        SectionTemplate("ROBOT-A", "ROBOT", "机器人创客 A", "T05", "R06", "Sun@10:50-12:20", 10),
        SectionTemplate("ROBOT-B", "ROBOT", "机器人创客 B", "T03", "R06", "Sat@15:50-17:20", 10),
        SectionTemplate("PHYS-A", "PHYS", "物理实验 A", "T05", "R07", "Thu@19:20-20:50", 10),
        SectionTemplate("PHYS-B", "PHYS", "物理实验 B", "T07", "R07", "Sun@15:50-17:20", 8),
        SectionTemplate("PIANO-A", "PIANO", "钢琴启蒙 A", "T04", "R05", "Sat@09:00-10:30", 6),
        SectionTemplate("PIANO-B", "PIANO", "钢琴启蒙 B", "T04", "R05", "Sun@09:00-10:30", 6),
        SectionTemplate("CALLI-A", "CALLI", "书法基础 A", "T06", "R08", "Fri@19:20-20:50", 10),
        SectionTemplate("CALLI-B", "CALLI", "书法基础 B", "T06", "R01", "Sun@10:50-12:20", 8),
    ]
    return {template.section_id: template for template in templates}


def _build_students() -> list[Student]:
    surnames = (
        "张", "王", "李", "赵", "陈", "刘", "杨", "黄", "周", "吴",
        "徐", "孙", "朱", "马", "胡", "郭", "何", "高", "林", "罗",
    )
    given_names = (
        "子涵", "雨桐", "浩然", "若溪", "景行", "安琪", "书睿", "嘉宁", "一诺", "梓轩",
        "语彤", "星宇", "可心", "晨曦", "逸辰", "沐言", "欣然", "铭泽", "楚涵", "思远",
        "乐彤", "嘉禾", "熙然", "宸宇", "知夏", "子墨", "锦程", "沛文", "皓轩", "宁安",
    )
    profiles = [
        (2, ("MATH", "ENG"), "周内晚间"),
        (3, ("MATH", "THINK"), "周二/周四"),
        (4, ("ENG", "CALLI"), "周末上午"),
        (4, ("PROG", "ROBOT"), "周末下午"),
        (5, ("MATH", "PROG"), "周内晚间"),
        (6, ("MATH", "PHYS"), "周末全天"),
        (2, ("PIANO", "ENG"), "周末上午"),
        (3, ("THINK", "ENG"), "周三/周六"),
        (5, ("ROBOT", "PHYS"), "周末下午"),
        (1, ("CALLI", "PIANO"), "周末上午"),
        (4, ("MATH", "ROBOT"), "周末优先"),
        (5, ("ENG", "PROG"), "工作日优先"),
    ]
    bonus_profiles = [
        (3, ("MATH", "ENG", "THINK"), "周末优先"),
        (5, ("MATH", "PROG"), "周内晚间"),
        (4, ("ROBOT", "PROG"), "周末下午"),
    ]
    students: list[Student] = []
    counter = 1
    for round_index in range(7):
        for profile in profiles:
            grade_level, selections, preference = profile
            name = surnames[(counter + round_index) % len(surnames)] + given_names[counter % len(given_names)]
            students.append(
                Student(
                    student_id=f"S{counter:03d}",
                    name=name,
                    grade_level=grade_level,
                    selections=selections,
                    preference_tag=preference,
                )
            )
            counter += 1
    for round_index in range(4):
        for profile in bonus_profiles:
            grade_level, selections, preference = profile
            name = surnames[(counter + round_index * 2) % len(surnames)] + given_names[(counter * 3) % len(given_names)]
            students.append(
                Student(
                    student_id=f"S{counter:03d}",
                    name=name,
                    grade_level=grade_level,
                    selections=selections,
                    preference_tag=preference,
                )
            )
            counter += 1
    return students


def _build_spec(
    dataset: DemoDataset,
    config: ScenarioConfig,
    strategy_key: str,
    template: SectionTemplate,
) -> SectionSpec:
    room = dataset.rooms[template.room_id]
    course = dataset.courses[template.course_code]
    capacity = min(template.base_capacity, room.capacity, config.class_size_cap)
    if strategy_key == "income" and course.price_per_student >= 1880:
        capacity = min(room.capacity, capacity + 1)
    elif strategy_key == "resource":
        capacity = min(room.capacity, capacity + 1)
    elif strategy_key == "open" and template.section_id in {"THINK-B", "PHYS-B", "CALLI-B"}:
        capacity = max(course.min_size, capacity - 1)
    return SectionSpec(
        section_id=template.section_id,
        course_code=template.course_code,
        label=template.label,
        teacher_id=template.teacher_id,
        room_id=template.room_id,
        slot_id=template.slot_id,
        capacity=capacity,
    )


def _assign_requests(
    dataset: DemoDataset,
    config: ScenarioConfig,
    section_specs: list[SectionSpec],
    strategy_key: str,
) -> tuple[list[ScheduledSection], list[EnrollmentAssignment], list[BlockedRequest]]:
    sections = [
        ScheduledSection(
            section_id=spec.section_id,
            course_code=spec.course_code,
            label=spec.label,
            teacher_id=spec.teacher_id,
            room_id=spec.room_id,
            slot_id=spec.slot_id,
            capacity=spec.capacity,
        )
        for spec in section_specs
    ]
    sections_by_course: dict[str, list[ScheduledSection]] = defaultdict(list)
    for section in sections:
        sections_by_course[section.course_code].append(section)

    student_map = {student.student_id: student for student in dataset.students}
    course_demand = dataset.demand_by_course
    requests: list[tuple[Student, str]] = []
    for student in dataset.students:
        for course_code in student.selections:
            requests.append((student, course_code))

    def request_sort_key(item: tuple[Student, str]) -> tuple[int, int, int]:
        student, course_code = item
        course = dataset.courses[course_code]
        if config.priority_mode == "高客单课程":
            return (-course.price_per_student, -course_demand[course_code], student.grade_level)
        if config.priority_mode == "低年级优先":
            return (student.grade_level, -course_demand[course_code], -course.price_per_student)
        if config.priority_mode == "周末黄金时段优先":
            return (-course_demand[course_code], -course.price_per_student, student.grade_level)
        return (-course_demand[course_code], -course.price_per_student, student.grade_level)

    requests.sort(key=request_sort_key)

    student_slots: dict[str, set[str]] = defaultdict(set)
    assignments: list[EnrollmentAssignment] = []
    blocked: list[BlockedRequest] = []

    for student, course_code in requests:
        candidates: list[tuple[int, float, ScheduledSection]] = []
        sections_for_course = sections_by_course.get(course_code, [])
        if not sections_for_course:
            blocked.append(
                BlockedRequest(
                    student_id=student.student_id,
                    student_name=student.name,
                    course_code=course_code,
                    course_name=dataset.courses[course_code].name,
                    reason="当前方案未为该课程开班",
                )
            )
            continue

        for section in sections_for_course:
            if section.assigned_count >= section.capacity:
                continue
            if section.slot_id in student_slots[student.student_id]:
                continue
            score = preference_match_score(student.preference_tag, section.slot_id)
            if config.priority_mode == "周末黄金时段优先" and section.slot_id in GOLDEN_SLOTS:
                score += 2
            if strategy_key == "resource":
                score += int(section.assigned_count / 3)
            else:
                score -= int(section.assigned_count / max(section.capacity, 1) * 2)
            spare_ratio = 1 - (section.assigned_count / max(section.capacity, 1))
            candidates.append((score, spare_ratio, section))

        if not candidates:
            same_course_sections = [section for section in sections_for_course if section.slot_id in student_slots[student.student_id]]
            reason = "与已选课程时间冲突" if same_course_sections else "课程班级容量不足"
            blocked.append(
                BlockedRequest(
                    student_id=student.student_id,
                    student_name=student.name,
                    course_code=course_code,
                    course_name=dataset.courses[course_code].name,
                    reason=reason,
                )
            )
            continue

        candidates.sort(key=lambda item: (-item[0], -item[1], item[2].slot_id))
        chosen = candidates[0][2]
        chosen.assigned_student_ids.append(student.student_id)
        student_slots[student.student_id].add(chosen.slot_id)
        assignments.append(
            EnrollmentAssignment(
                student_id=student.student_id,
                student_name=student.name,
                course_code=course_code,
                course_name=dataset.courses[course_code].name,
                section_id=chosen.section_id,
                slot_id=chosen.slot_id,
            )
        )

    return sections, assignments, blocked


def _cancel_under_enrolled_sections(
    dataset: DemoDataset,
    config: ScenarioConfig,
    sections: list[ScheduledSection],
    strategy_key: str,
) -> tuple[list[ScheduledSection], list[EnrollmentAssignment], list[BlockedRequest]]:
    under_enrolled_ids = {
        section.section_id
        for section in sections
        if section.assigned_count > 0
        and section.assigned_count < dataset.courses[section.course_code].min_size
    }
    if not under_enrolled_ids:
        assignments, blocked = _rebuild_assignments_from_sections(dataset, sections, config, strategy_key)
        return sections, assignments, blocked

    rebuilt_specs = [
        SectionSpec(
            section_id=section.section_id,
            course_code=section.course_code,
            label=section.label,
            teacher_id=section.teacher_id,
            room_id=section.room_id,
            slot_id=section.slot_id,
            capacity=section.capacity,
        )
        for section in sections
        if section.section_id not in under_enrolled_ids
    ]
    rebuilt_sections, assignments, blocked = _assign_requests(dataset, config, rebuilt_specs, strategy_key)
    for cancelled_id in under_enrolled_ids:
        original = next(section for section in sections if section.section_id == cancelled_id)
        rebuilt_sections.append(
            ScheduledSection(
                section_id=original.section_id,
                course_code=original.course_code,
                label=original.label,
                teacher_id=original.teacher_id,
                room_id=original.room_id,
                slot_id=original.slot_id,
                capacity=original.capacity,
                assigned_student_ids=[],
                status="CANCELLED",
                note="报名人数低于成班下限，已转入候补。",
            )
        )
        blocked.extend(
            BlockedRequest(
                student_id=student_id,
                student_name=next(student.name for student in dataset.students if student.student_id == student_id),
                course_code=original.course_code,
                course_name=dataset.courses[original.course_code].name,
                reason="所在班级未达到成班下限",
            )
            for student_id in original.assigned_student_ids
        )
    rebuilt_sections.sort(key=lambda section: section.slot_id)
    return rebuilt_sections, assignments, blocked


def _rebuild_assignments_from_sections(
    dataset: DemoDataset,
    sections: list[ScheduledSection],
    config: ScenarioConfig,
    strategy_key: str,
) -> tuple[list[EnrollmentAssignment], list[BlockedRequest]]:
    specs = [
        SectionSpec(
            section_id=section.section_id,
            course_code=section.course_code,
            label=section.label,
            teacher_id=section.teacher_id,
            room_id=section.room_id,
            slot_id=section.slot_id,
            capacity=section.capacity,
        )
        for section in sections
        if section.status == "OPENED"
    ]
    refreshed_sections, assignments, blocked = _assign_requests(dataset, config, specs, strategy_key)
    assignment_map = {section.section_id: section.assigned_student_ids for section in refreshed_sections}
    for section in sections:
        section.assigned_student_ids = assignment_map.get(section.section_id, [])
    return assignments, blocked


def _build_metrics(
    dataset: DemoDataset,
    sections: list[ScheduledSection],
    assignments: list[EnrollmentAssignment],
    blocked: list[BlockedRequest],
    config: ScenarioConfig,
) -> PlanMetrics:
    active_sections = [section for section in sections if section.status == "OPENED"]
    active_teacher_ids = {section.teacher_id for section in active_sections}
    active_room_ids = {section.room_id for section in active_sections}
    teacher_capacity = sum(dataset.teachers[teacher_id].target_sections for teacher_id in active_teacher_ids)
    room_capacity = sum(dataset.rooms[room_id].target_slots for room_id in active_room_ids)
    revenue = sum(
        dataset.courses[section.course_code].price_per_student * section.assigned_count
        for section in active_sections
    )
    fill_rates = [section.assigned_count / max(section.capacity, 1) for section in active_sections]
    golden_count = sum(1 for section in active_sections if section.slot_id in GOLDEN_SLOTS)
    golden_divisor = 8 if config.add_evening_slot else 6

    return PlanMetrics(
        revenue=revenue,
        assigned_requests=len(assignments),
        total_requests=dataset.total_requests,
        open_classes=len(active_sections),
        candidate_classes=len(sections),
        teacher_utilization=min(0.99, len(active_sections) / max(teacher_capacity, 1)),
        room_utilization=min(0.99, len(active_sections) / max(room_capacity, 1)),
        golden_utilization=min(0.99, golden_count / golden_divisor),
        fill_rate=mean(fill_rates) if fill_rates else 0.0,
        unassigned_requests=len(blocked),
    )


def _build_course_breakdown(
    dataset: DemoDataset,
    sections: list[ScheduledSection],
    blocked: list[BlockedRequest],
) -> list[dict[str, object]]:
    active_sections = [section for section in sections if section.status == "OPENED"]
    sections_by_course: Counter[str] = Counter(section.course_code for section in active_sections)
    assigned_by_course: Counter[str] = Counter()
    revenue_by_course: Counter[str] = Counter()
    for section in active_sections:
        assigned_by_course[section.course_code] += section.assigned_count
        revenue_by_course[section.course_code] += (
            section.assigned_count * dataset.courses[section.course_code].price_per_student
        )
    blocked_by_course: Counter[str] = Counter(item.course_code for item in blocked)

    rows: list[dict[str, object]] = []
    for course_code, demand in dataset.demand_by_course.most_common():
        course = dataset.courses[course_code]
        rows.append(
            {
                "course_code": course_code,
                "course_name": course.name,
                "demand": demand,
                "assigned": assigned_by_course[course_code],
                "blocked": blocked_by_course[course_code],
                "sections": sections_by_course[course_code],
                "revenue": revenue_by_course[course_code],
                "room_type": " / ".join(ROOM_TYPE_LABELS[item] for item in course.room_types),
            }
        )
    return rows


def _build_conflicts(
    dataset: DemoDataset,
    config: ScenarioConfig,
    sections: list[ScheduledSection],
    blocked: list[BlockedRequest],
) -> list[Conflict]:
    conflicts: list[Conflict] = []
    active_sections = [section for section in sections if section.status == "OPENED"]
    for section in sections:
        if section.status == "CANCELLED":
            course = dataset.courses[section.course_code]
            conflicts.append(
                Conflict(
                    severity="高",
                    title=f"{course.name} 未成班",
                    detail=f"{section.label} 报名人数低于成班下限，当前已转入候补。",
                    suggestion="可尝试合并同课程班、放宽低开规则或追加周末时段。",
                )
            )

    teacher_slot_seen: dict[tuple[str, str], str] = {}
    room_slot_seen: dict[tuple[str, str], str] = {}
    for section in active_sections:
        course = dataset.courses[section.course_code]
        teacher = dataset.teachers[section.teacher_id]
        room = dataset.rooms[section.room_id]
        if course.teacher_skill not in teacher.skills:
            conflicts.append(
                Conflict(
                    severity="高",
                    title="教师资质不匹配",
                    detail=f"{teacher.name} 当前被安排到 {course.name}，但资质标签不匹配。",
                    suggestion="更换为具备同课程资质的教师，或锁定本班避免手工调乱。",
                )
            )
        if room.room_type not in course.room_types:
            conflicts.append(
                Conflict(
                    severity="高",
                    title="教室类型不匹配",
                    detail=f"{section.label} 使用 {room.name}，与 {course.name} 所需教室类型不一致。",
                    suggestion="切回匹配教室类型，或为该课程新增专用教室。",
                )
            )
        if section.slot_id not in teacher.availability:
            conflicts.append(
                Conflict(
                    severity="中",
                    title="教师可用时段冲突",
                    detail=f"{teacher.name} 并未在 {slot_to_label(section.slot_id)} 标记为可授课。",
                    suggestion="优先调整到教师可用时段，避免方案落地时二次改课。",
                )
            )
        if section.slot_id not in room.availability:
            conflicts.append(
                Conflict(
                    severity="中",
                    title="教室时段不可用",
                    detail=f"{room.name} 在 {slot_to_label(section.slot_id)} 不在开放时段内。",
                    suggestion="切换到可用教室，或在沙盘中模拟新增开放时段。",
                )
            )
        teacher_key = (section.teacher_id, section.slot_id)
        room_key = (section.room_id, section.slot_id)
        if teacher_key in teacher_slot_seen:
            conflicts.append(
                Conflict(
                    severity="高",
                    title="教师重复占用",
                    detail=(
                        f"{teacher.name} 同时承担 {teacher_slot_seen[teacher_key]} 与 {section.label}，"
                        f"均在 {slot_to_label(section.slot_id)}。"
                    ),
                    suggestion="人工调度页中更换教师或改班时段。",
                )
            )
        else:
            teacher_slot_seen[teacher_key] = section.label
        if room_key in room_slot_seen:
            conflicts.append(
                Conflict(
                    severity="高",
                    title="教室重复占用",
                    detail=(
                        f"{room.name} 同时排入 {room_slot_seen[room_key]} 与 {section.label}，"
                        f"均在 {slot_to_label(section.slot_id)}。"
                    ),
                    suggestion="更换教室或为其中一个班级切换时段。",
                )
            )
        else:
            room_slot_seen[room_key] = section.label

    blocked_by_reason = Counter(item.reason for item in blocked)
    for reason, count in blocked_by_reason.most_common(2):
        conflicts.append(
            Conflict(
                severity="中",
                title="未满足需求仍较多",
                detail=f"{count} 个选课请求未被满足，主要原因是：{reason}。",
                suggestion="优先针对该瓶颈追加时段或资源，再比较重算后的新方案。",
            )
        )

    deduped: list[Conflict] = []
    seen = set()
    for conflict in conflicts:
        signature = (conflict.title, conflict.detail)
        if signature not in seen:
            deduped.append(conflict)
            seen.add(signature)
    return deduped[:10]


def _build_generation_notes(
    dataset: DemoDataset,
    config: ScenarioConfig,
    strategy_key: str,
    sections: list[ScheduledSection],
    blocked: list[BlockedRequest],
) -> list[str]:
    active_sections = [section for section in sections if section.status == "OPENED"]
    demand = dataset.total_requests
    notes = [
        f"读取学生 {len(dataset.students)} 人，累计选课请求 {demand} 条。",
        f"当前策略为 {STRATEGY_LABELS[strategy_key]}，共生成候选班级 {len(sections)} 个。",
        f"成功开班 {len(active_sections)} 个，未满足请求 {len(blocked)} 条。",
    ]
    if config.add_math_teacher:
        notes.append("已纳入新增数学教师，数学与思维课程容量上浮。")
    if config.add_small_room:
        notes.append("已纳入新增小班教室，英语与思维课程可增加晚高峰承载。")
    if config.add_evening_slot:
        notes.append("已开放周末晚间时段，用于承接编程与思维课程的溢出需求。")
    if config.lock_key_classes:
        notes.append("重点班级已锁定在高满意度时段，优先保证示范班稳定性。")
    return notes


def _build_recommendations(
    dataset: DemoDataset,
    config: ScenarioConfig,
    sections: list[ScheduledSection],
    blocked: list[BlockedRequest],
) -> list[str]:
    recommendations: list[str] = []
    blocked_by_course = Counter(item.course_code for item in blocked)
    if blocked_by_course:
        top_course, top_count = blocked_by_course.most_common(1)[0]
        course = dataset.courses[top_course]
        recommendations.append(
            f"{course.name} 仍有 {top_count} 条未满足请求，建议优先围绕该课程追加容量。"
        )
        if top_course in {"MATH", "THINK"} and not config.add_math_teacher:
            recommendations.append("当前数学/思维瓶颈偏向师资，建议在沙盘中模拟新增 1 名数学老师。")
        if top_course in {"ENG", "MATH", "THINK"} and not config.add_small_room:
            recommendations.append("当前小班教室偏紧，新增一间小班教室对高峰时段最有效。")
        if top_course in {"PROG", "ROBOT"} and not config.add_evening_slot:
            recommendations.append("编程类课程更依赖周末高峰，开放周末晚间时段会更直接提升承载。")
    else:
        recommendations.append("当前方案已较好覆盖需求，下一步可重点比较收入与资源压力差异。")

    active_sections = [section for section in sections if section.status == "OPENED"]
    golden_count = sum(1 for section in active_sections if section.slot_id in GOLDEN_SLOTS)
    if golden_count < 5:
        recommendations.append("黄金时段利用偏低，可将高需求课程进一步向周末上午集中。")
    if any(section.status == "CANCELLED" for section in sections):
        recommendations.append("存在低开未成班课程，建议内部讨论是否允许低开或合并相邻班型。")
    return recommendations[:4]


def _room_label(dataset: DemoDataset, room_id: str) -> str:
    room = dataset.rooms[room_id]
    return f"{room.name} ({ROOM_TYPE_LABELS[room.room_type]})"


def room_options(dataset: DemoDataset, include_small_room: bool) -> list[str]:
    room_ids = [room_id for room_id in dataset.rooms if include_small_room or room_id != "R09"]
    return [room_id for room_id in room_ids]


def teacher_options(dataset: DemoDataset, include_math_teacher: bool) -> list[str]:
    teacher_ids = [teacher_id for teacher_id in dataset.teachers if include_math_teacher or teacher_id != "T11"]
    return [teacher_id for teacher_id in teacher_ids]
