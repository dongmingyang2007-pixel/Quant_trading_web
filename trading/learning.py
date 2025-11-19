from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional
import json

from django.conf import settings


@dataclass(frozen=True, slots=True)
class ModuleSection:
    title: str
    summary: str
    body: list[str]
    topics: list[str]
    image: str | None = None
    image_caption: str | None = None
    anchor: str | None = None


@dataclass(frozen=True, slots=True)
class ModuleCaseStudy:
    title: str
    scenario: str
    discussion: list[str]
    takeaway: str


@dataclass(frozen=True, slots=True)
class ModuleAssignment:
    title: str
    description: str
    deliverables: list[str]
    checkpoints: list[str]


@dataclass(frozen=True, slots=True)
class LearningModule:
    slug: str
    title: str
    level: str
    duration: str
    summary: str
    cover_image: str | None
    objectives: list[str]
    milestones: list[dict[str, str]]
    sections: list[ModuleSection]
    case_studies: list[ModuleCaseStudy]
    assignments: list[ModuleAssignment]
    toolkit: list[str]
    reading_notes: list[str]
    resources: list[dict[str, str]]
    call_to_action: str


@dataclass(frozen=True, slots=True)
class MicroLesson:
    title: str
    focus: str
    action: str


@dataclass(frozen=True, slots=True)
class SoftwareGuide:
    title: str
    summary: str
    steps: list[str]
    link_label: str | None = None
    link_target: str | None = None


@dataclass(frozen=True, slots=True)
class DataClinic:
    title: str
    focus: str
    checklist: list[str]
    reminder: str


@dataclass(frozen=True, slots=True)
class PracticeSprint:
    title: str
    duration: str
    objectives: list[str]
    deliverable: str
    link_label: str | None = None
    link_target: str | None = None


@dataclass(frozen=True, slots=True)
class LearningMetric:
    label: str
    value: str
    hint: str


@dataclass(frozen=True, slots=True)
class TopicNavigator:
    slug: str
    title: str
    strapline: str
    summary: str
    level: str
    duration: str
    link_target: str
    color: str


@dataclass(frozen=True, slots=True)
class CollectionCourse:
    title: str
    summary: str
    level: str
    duration: str
    anchor: str
    badge: str | None = None


@dataclass(frozen=True, slots=True)
class CourseCollection:
    slug: str
    title: str
    description: str
    theme: str
    courses: list[CollectionCourse]


@dataclass(frozen=True, slots=True)
class LearningFAQ:
    question: str
    answer: str


@dataclass(frozen=True, slots=True)
class FilterGroup:
    title: str
    items: list[dict[str, str]]


_CONTENT_DIR = settings.LEARNING_CONTENT_DIR
_INDEX_FILE = _CONTENT_DIR / "index.json"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_section(raw: dict) -> ModuleSection:
    return ModuleSection(
        title=raw.get("title", ""),
        summary=raw.get("summary", ""),
        body=list(raw.get("body", [])),
        topics=list(raw.get("topics", [])),
        image=raw.get("image"),
        image_caption=raw.get("image_caption"),
        anchor=raw.get("anchor"),
    )


def _build_case(raw: dict) -> ModuleCaseStudy:
    return ModuleCaseStudy(
        title=raw.get("title", ""),
        scenario=raw.get("scenario", ""),
        discussion=list(raw.get("discussion", [])),
        takeaway=raw.get("takeaway", ""),
    )


def _build_assignment(raw: dict) -> ModuleAssignment:
    return ModuleAssignment(
        title=raw.get("title", ""),
        description=raw.get("description", ""),
        deliverables=list(raw.get("deliverables", [])),
        checkpoints=list(raw.get("checkpoints", [])),
    )


def _build_module(raw: dict) -> LearningModule:
    sections = [_build_section(item) for item in raw.get("sections", [])]
    cases = [_build_case(item) for item in raw.get("case_studies", [])]
    assignments = [_build_assignment(item) for item in raw.get("assignments", [])]
    return LearningModule(
        slug=raw.get("slug", ""),
        title=raw.get("title", ""),
        level=raw.get("level", ""),
        duration=raw.get("duration", ""),
        summary=raw.get("summary", ""),
        cover_image=raw.get("cover_image"),
        objectives=list(raw.get("objectives", [])),
        milestones=list(raw.get("milestones", [])),
        sections=sections,
        case_studies=cases,
        assignments=assignments,
        toolkit=list(raw.get("toolkit", [])),
        reading_notes=list(raw.get("reading_notes", [])),
        resources=list(raw.get("resources", [])),
        call_to_action=raw.get("call_to_action", ""),
    )


@lru_cache(maxsize=1)
def _load_modules() -> list[LearningModule]:
    if not _INDEX_FILE.exists():
        return []
    index = _load_json(_INDEX_FILE)
    modules: list[LearningModule] = []
    for entry in index:
        slug = entry.get("slug")
        filename = entry.get("file")
        if not slug or not filename:
            continue
        path = _CONTENT_DIR / filename
        if not path.exists():
            continue
        data = _load_json(path)
        modules.append(_build_module(data))
    return modules


def get_learning_tracks() -> list[LearningModule]:
    return list(_load_modules())


def get_learning_track(slug: str) -> Optional[LearningModule]:
    for module in _load_modules():
        if module.slug == slug:
            return module
    return None


LEARNING_METRICS: list[LearningMetric] = [
    LearningMetric(label="课程结构", value="1 门专题 · 32 节内容", hint="含引言、学习目标与 11 个章节下的细分小节，覆盖假设写作、显著性水平、z/t 检验和比例检验。"),
    LearningMetric(label="建议学时", value="≈ 6 小时", hint="建议按“概念 → 算例 → 实作”三段完成学习。"),
    LearningMetric(label="配套资源", value="4 套模板", hint="附 Excel/ Python 示例与沟通模板，方便落地。"),
]


TOPIC_NAVIGATORS: list[TopicNavigator] = [
    TopicNavigator(
        slug="hypothesis-testing",
        title="假设检验专题",
        strapline="用统计判定商业假设",
        summary="写出 H0/H1、设置显著性、执行 z/t 检验，并将结果翻译成可执行的商业语言。",
        level="入门",
        duration="建议 6 小时",
        link_target="hypothesis-testing",
        color="rose"
    )
]


COURSE_COLLECTIONS: list[CourseCollection] = [
    CourseCollection(
        slug="pathway-hypothesis",
        title="学习路线：假设检验",
        description="按“概念 → 显著性 → 检验 → 实作”的顺序安排行程，每一步都有对接的章节与练习。",
        theme="rose",
        courses=[
            CollectionCourse(title="写出可检验的假设", summary="掌握 H0/H1 写法与检验方向的选择。", level="基础", duration="60 分钟", anchor="chapter-1-hypothesis-testing"),
            CollectionCourse(title="显著性与拒绝域", summary="理解 \\(\\alpha\\)、临界值与 p 值的互动关系。", level="进阶", duration="90 分钟", anchor="chapter-2-alpha-levels"),
            CollectionCourse(title="z/t 检验实战", summary="完成均值与比例的 z 检验，再以 t 检验复验样本。", level="实践", duration="120 分钟", anchor="chapter-4-z-vs-t"),
            CollectionCourse(title="作业与反思", summary="将检验结果写成业务说明，并规划后续迭代。", level="巩固", duration="60 分钟", anchor="chapter-10-reflection", badge="建议收尾")
        ],
    )
]


MICRO_LESSONS: list[MicroLesson] = [
    MicroLesson(title="识别假设类型", focus="区分均值假设与比例假设，并判断单尾/双尾方向。", action="选一个业务问题，写下 H0/H1，并注明显著性与检验方向。"),
    MicroLesson(title="计算显著性水平", focus="把置信水平转换为 \\(\\alpha\\) 并在图表上画出拒绝域。", action="在 Excel 中建立‘置信水平→显著性→临界值’的小表格，并截图保存。"),
    MicroLesson(title="z 检验四步演练", focus="从样本均值到 z 值，再到临界值与结论。", action="使用提供的样本数据，完成一次 z 检验并写下是否拒绝 H0。"),
    MicroLesson(title="p 值解释卡片", focus="把 p 值翻译成自然语言，让非统计背景也能理解。", action="针对一条检验结果写三行说明：统计结论、概率解释、行动建议。"),
    MicroLesson(title="t 检验自由度提示", focus="理解自由度对临界值的影响。", action="建立一张自由度与临界值对照表，观察 df 变化时临界值的趋势。")
]


SOFTWARE_GUIDES: list[SoftwareGuide] = [
    SoftwareGuide(
        title="Excel 完成一次 z 检验",
        summary="使用模板函数快速计算均值、标准差、z 值与 p 值。",
        steps=[
            "在数据表中输入样本，使用 `AVERAGE`、`STDEV.P` 或 `STDEV.S` 计算统计量。",
            "通过 `NORM.S.INV` 获得临界值，并用 `NORM.S.DIST` 计算 p 值。",
            "把结果写入结论区域：是否拒绝 H0、差异方向、行动建议。"
        ],
        link_label="下载 Excel 模板",
        link_target="hypothesis-testing"
    ),
    SoftwareGuide(
        title="Python 复现 z/t 检验",
        summary="借助 pandas 与 SciPy，在脚本中完成假设检验并输出报告。",
        steps=[
            "读取 CSV 或数据库样本，使用 pandas 计算均值与标准差。",
            "调用 `scipy.stats` 中的 `ttest_1samp` 或自写 z 检验函数，获取统计量与 p 值。",
            "利用 Jupyter Notebook 或 Markdown 模板生成结论说明，记录输入参数和输出结果。"
        ],
        link_label="查看 Python 示例",
        link_target="hypothesis-testing"
    )
]


DATA_CLINICS: list[DataClinic] = [
    DataClinic(
        title="样本采集体检",
        focus="确保抽样方法、时间窗口与数据清洗满足检验要求。",
        checklist=[
            "确认样本是否独立、随机，记录采集时间与排除规则。",
            "检查缺失值与异常值处理方式，并说明是否会影响检验结果。",
            "核对样本量是否满足 \\(np_0\\) 与 \\(n(1-p_0) > 5\\) 或 t 检验最小样本量要求。"
        ],
        reminder="数据质量决定检验可信度，遇到疑点先解决数据再做统计。"
    ),
    DataClinic(
        title="显著性决策表",
        focus="把显著性水平、临界值、拒绝域和业务含义集中记录。",
        checklist=[
            "列出不同 \\(\\alpha\\) 下的临界值，并标明适用的业务场景。",
            "说明拒绝 H0 的行动方案，以及保持 H0 时的监控措施。",
            "记录曾经使用的 \\(\\alpha\\) 与结果，为后续复检提供参考。"
        ],
        reminder="显著性选择应可追溯，避免出现‘结果出来才谈阈值’的争议。"
    ),
    DataClinic(
        title="检验结果复核",
        focus="确保统计结论、代码/公式与业务解读一致。",
        checklist=[
            "让同事复算一次统计量与 p 值，确认计算无误。",
            "对照假设写法，检查检验方向是否一致（单尾/双尾）。",
            "检查报告中是否说明样本来源、显著性和下一步行动。"
        ],
        reminder="所有结论都应能被复现与复核，才能在团队内部建立信任。"
    )
]


PRACTICE_SPRINTS: list[PracticeSprint] = [
    PracticeSprint(
        title="60 分钟：完成一次 z 检验",
        duration="60 分钟",
        objectives=[
            "选择一个均值问题，收集 30 条以上样本数据并记录在模板中。",
            "按照六步法计算 z 值与 p 值，判断是否拒绝 H0。",
            "用 200 字写出结论、影响与后续行动。"
        ],
        deliverable="上传含有计算截图与结论说明的文档。",
        link_label="打开检验模板",
        link_target="hypothesis-testing"
    ),
    PracticeSprint(
        title="90 分钟：t 检验 + 比例检验组合练习",
        duration="90 分钟",
        objectives=[
            "使用抽样数据完成一次单样本 t 检验，并记录自由度、p 值。",
            "使用分类数据完成一次比例检验，验证占比是否达到目标。",
            "把两个检验的结论整理成一个汇总报告，给出行动建议。"
        ],
        deliverable="提交 Excel 或 Notebook 文件与汇总报告。",
        link_label="下载练习数据",
        link_target="hypothesis-testing"
    )
]


LEARNING_FAQS: list[LearningFAQ] = [
    LearningFAQ(question="需要先学哪些数学或统计知识？", answer="掌握均值、标准差、正态分布等基础概念即可。本专题会在每个环节复习必要公式，并提供可直接使用的模板。"),
    LearningFAQ(question="什么时候用 z 检验、什么时候用 t 检验？", answer="如果已知总体标准差或样本量很大，可使用 z 检验；若只能用样本估计标准差，就使用 t 检验。本课程提供决策流程图，帮助快速判断。"),
    LearningFAQ(question="检验结果需要保存哪些资料？", answer="请保留假设说明、显著性水平、样本数据、计算过程（公式或代码）以及结论，这些元素缺一不可。"),
    LearningFAQ(question="如何向非统计背景的同事解释 p 值？", answer="可以说“如果原假设正确，出现当前或更极端结果的概率是 X%”，并补充行动建议，例如是否需要调整策略或继续观察。")
]


LEARNING_FILTER_GROUPS: list[FilterGroup] = [
    FilterGroup(
        title="学习阶段",
        items=[
            {"label": "概念理解", "description": "阅读假设、显著性与检验类型的核心概念。"},
            {"label": "公式演练", "description": "跟随示例完成 z/t 检验与比例检验的计算。"},
            {"label": "实作报告", "description": "将检验结果整理成业务语言并输出行动建议。"}
        ]
    ),
    FilterGroup(
        title="工具选择",
        items=[
            {"label": "Excel 模板", "description": "使用现成函数完成单次检验，快速输出结果。"},
            {"label": "Python 脚本", "description": "适合批量检测与自动化报表。"},
            {"label": "沟通模板", "description": "帮助把统计术语翻译成业务可以理解的语言。"}
        ]
    ),
    FilterGroup(
        title="应用场景",
        items=[
            {"label": "营销实验", "description": "检验转化率、点击率等比例指标是否达标。"},
            {"label": "运营效率", "description": "评估响应时间、处理时长等均值指标是否改进。"},
            {"label": "产品试点", "description": "在小样本试点中验证改版效果，再决定是否推广。"}
        ]
    )
]


def get_learning_metrics() -> list[LearningMetric]:
    return LEARNING_METRICS


def get_topic_navigators() -> list[TopicNavigator]:
    return TOPIC_NAVIGATORS


def get_course_collections() -> list[CourseCollection]:
    return COURSE_COLLECTIONS


def get_micro_lessons() -> list[MicroLesson]:
    return MICRO_LESSONS


def get_software_guides() -> list[SoftwareGuide]:
    return SOFTWARE_GUIDES


def get_data_clinics() -> list[DataClinic]:
    return DATA_CLINICS


def get_practice_sprints() -> list[PracticeSprint]:
    return PRACTICE_SPRINTS


def get_learning_faqs() -> list[LearningFAQ]:
    return LEARNING_FAQS


def get_filter_groups() -> list[FilterGroup]:
    return LEARNING_FILTER_GROUPS
