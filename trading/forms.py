from datetime import date

from django import forms
from django.conf import settings
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.utils.translation import get_language


class QuantStrategyForm(forms.Form):
    ML_MODE_CHOICES_ZH = [
        ("light", "轻量：梯度提升 (默认)"),
        ("deep", "深度：LSTM 序列模型"),
        ("transformer", "深度：Transformer 编码器"),
        ("fusion", "深度：自动融合 LSTM + Transformer"),
    ]
    ML_MODE_CHOICES_EN = [
        ("light", "Light · Gradient Boosting (default)"),
        ("deep", "Deep · LSTM sequence model"),
        ("transformer", "Deep · Transformer encoder"),
        ("fusion", "Deep · Fusion (auto LSTM + Transformer)"),
    ]

    ticker = forms.CharField(
        max_length=16,
        label="主要标的代码",
        help_text="支持 Yahoo Finance 代码，例如 NVDA、AAPL、600519.SS。",
    )
    benchmark_ticker = forms.CharField(
        max_length=16,
        required=False,
        label="对比基准（可选）",
        help_text="默认对比 SPY。若专注单一资产可留空。",
    )
    start_date = forms.DateField(
        label="开始日期",
        widget=forms.DateInput(attrs={"type": "date"}),
    )
    end_date = forms.DateField(
        label="结束日期",
        widget=forms.DateInput(attrs={"type": "date"}),
    )
    capital = forms.DecimalField(
        min_value=0,
        required=False,
        initial=250000,
        label="模拟资金规模",
        help_text="用于资产配置建模，默认 250,000。",
    )
    ml_mode = forms.ChoiceField(
        label="模型类型",
        choices=ML_MODE_CHOICES_ZH,
        initial="light",
        help_text="轻量 GBDT 兼具速度与稳定性；深度模型可捕捉更多时序模式，Fusion 会自动融合 LSTM 与 Transformer。",
    )

    def __init__(self, *args, language=None, **kwargs):
        self.language = (language or get_language() or "").lower()
        self.lang_is_zh = self.language.startswith("zh")
        super().__init__(*args, **kwargs)
        def _msg(en: str, zh: str) -> str:
            return zh if self.lang_is_zh else en
        self._msg = _msg
        label_map = {
            "ticker": ("Primary ticker", "主要标的代码"),
            "benchmark_ticker": ("Benchmark (optional)", "对比基准（可选）"),
            "start_date": ("Start date", "开始日期"),
            "end_date": ("End date", "结束日期"),
            "capital": ("Capital base", "模拟资金规模"),
            "ml_mode": ("Model type", "模型类型"),
        }
        help_map = {
            "ticker": (
                "Accepts Yahoo Finance symbols, e.g., NVDA, AAPL, 600519.SS.",
                "支持 Yahoo Finance 代码，例如 NVDA、AAPL、600519.SS。",
            ),
            "benchmark_ticker": (
                "Defaults to SPY. Leave blank when focusing on a single asset.",
                "默认对比 SPY。若专注单一资产可留空。",
            ),
            "capital": (
                "Used for allocation modeling. Default 250,000.",
                "用于资产配置建模，默认 250,000。",
            ),
            "ml_mode": (
                "Light GBDT balances speed and stability; deep modes capture richer sequences. Fusion blends LSTM and Transformer automatically.",
                "轻量 GBDT 兼具速度与稳定性；深度模型可捕捉更多时序模式；Fusion 会自动融合 LSTM 与 Transformer。",
            ),
        }
        for field, labels in label_map.items():
            self.fields[field].label = labels[1] if self.lang_is_zh else labels[0]
        for field, helps in help_map.items():
            self.fields[field].help_text = helps[1] if self.lang_is_zh else helps[0]
        self.fields["ml_mode"].choices = self.ML_MODE_CHOICES_ZH if self.lang_is_zh else self.ML_MODE_CHOICES_EN
        self.warnings: list[str] = []
        self.fields["ticker"].widget.attrs.update({"placeholder": "NVDA"})
        self.fields["benchmark_ticker"].widget.attrs.update({"placeholder": "SPY"})
        for name in ("start_date", "end_date", "capital"):
            self.fields[name].widget.attrs.setdefault("class", "form-control")
        for name in ("ticker", "benchmark_ticker", "ml_mode"):
            self.fields[name].widget.attrs.setdefault("class", "form-control")

    def clean(self):
        cleaned = super().clean()
        start = cleaned.get("start_date")
        end = cleaned.get("end_date")
        if start and end:
            today = date.today()
            if end > today:
                cleaned["end_date"] = today
                self.warnings.append(
                    self._msg(
                        f"End date adjusted to {today} to avoid using future data.",
                        f"结束日期已自动校正为 {today}，避免使用未来数据。",
                    )
                )
                end = today
            if start >= end:
                raise ValidationError(self._msg("Start date must be earlier than end date.", "开始日期必须早于结束日期。"))
            if (end - start).days < 120:
                self.warnings.append(
                    self._msg(
                        "The backtest window is shorter than 120 days. Extend it for more stable statistics.",
                        "回测窗口少于 120 天，建议拉长周期以提高统计稳定性。",
                    )
                )
        ticker = (cleaned.get("ticker") or "").strip().upper()
        benchmark = (cleaned.get("benchmark_ticker") or "").strip().upper()
        if benchmark and benchmark == ticker:
            self.warnings.append(
                self._msg(
                    "Benchmark matches the primary ticker, so it was ignored.",
                    "基准代码与主要标的一致，已自动忽略基准。",
                )
            )
            cleaned["benchmark_ticker"] = ""
        capital = cleaned.get("capital")
        if capital is not None and capital < 10000:
            self.warnings.append(
                self._msg(
                    "Capital below 10,000 increases the impact of commissions and minimum lot sizes.",
                    "模拟资金低于 10,000，实际交易时需关注手续费与最小交易单位。",
                )
            )
        return cleaned


class ProfileForm(forms.Form):
    display_name = forms.CharField(
        max_length=40,
        required=False,
        label="展示昵称",
        help_text="显示在个人主页上的名称，默认沿用账号用户名。",
    )
    cover_color = forms.CharField(
        max_length=16,
        required=False,
        initial="#116e5f",
        label="背景主色",
        help_text="",
        widget=forms.TextInput(attrs={"type": "color", "class": "form-control form-control-color"}),
    )
    bio = forms.CharField(
        required=False,
        label="个人简介",
        widget=forms.Textarea(attrs={"rows": 3, "placeholder": "介绍一下你的投资风格或兴趣..."}),
    )
    avatar = forms.ImageField(
        required=False,
        label="上传头像",
        help_text="支持 JPG/PNG，建议尺寸 400x400 以上。",
    )
    feature_image = forms.ImageField(
        required=False,
        label="展示照片",
        help_text="可选，用于展示在主页的代表性照片。",
    )
    feature_cropped_data = forms.CharField(required=False, widget=forms.HiddenInput)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("display_name", "bio"):
            self.fields[name].widget.attrs.setdefault("class", "form-control")
        cover_widget = self.fields["cover_color"].widget
        cover_widget.attrs.setdefault("class", "form-control form-control-color")
        # Avatar uses regular form-control styling
        self.fields["avatar"].widget.attrs.setdefault("class", "form-control")
        # Featured image is hidden and controlled by JS uploader
        self.fields["feature_image"].widget.attrs.update(
            {
                "class": "d-none",
                "data-role": "feature-input",
                "accept": "image/*",
            }
        )
        self.fields["feature_cropped_data"].widget.attrs.update(
            {
                "data-role": "feature-cropped",
            }
        )


class CommunityPostForm(forms.Form):
    content = forms.CharField(
        label="想法",
        widget=forms.Textarea(attrs={"rows": 3, "placeholder": "分享你的策略心得、市场观察或经验..."}),
    )
    topic = forms.ChoiceField(
        label="话题",
        required=False,
        choices=[],
    )
    new_topic_name = forms.CharField(
        label="新话题名称",
        max_length=60,
        required=False,
        widget=forms.TextInput(attrs={"placeholder": "创建新话题…"}),
    )
    new_topic_description = forms.CharField(
        label="话题介绍",
        required=False,
        widget=forms.Textarea(attrs={"rows": 2, "placeholder": "描述话题的讨论方向…"}),
    )
    image = forms.ImageField(
        required=False,
        label="图片上传",
        help_text="可选，支持 JPG/PNG。",
    )
    image_cropped_data = forms.CharField(
        required=False,
        widget=forms.HiddenInput,
    )

    def __init__(self, *args, topics: list[tuple[str, str]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["content"].widget.attrs.setdefault("class", "form-control")
        self.fields["topic"].widget.attrs.setdefault("class", "form-select")
        self.fields["new_topic_name"].widget.attrs.setdefault("class", "form-control")
        self.fields["new_topic_description"].widget.attrs.setdefault("class", "form-control")
        self.fields["image"].widget.attrs.setdefault("class", "form-control")
        if topics:
            self.fields["topic"].choices = topics
        else:
            self.fields["topic"].choices = []


class SignupForm(UserCreationForm):
    email = forms.EmailField(label="邮箱", help_text="用于找回密码与通知", required=True)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email")

    def __init__(self, *args, language=None, **kwargs):
        self.language = (language or get_language() or "").lower()
        super().__init__(*args, **kwargs)
        is_zh = self.language.startswith("zh")
        self.fields["email"].label = "邮箱" if is_zh else "Email"
        self.fields["email"].help_text = "用于找回密码与通知" if is_zh else "Used for password reset and notifications."

    def clean_email(self):
        email = (self.cleaned_data.get("email") or "").strip()
        if not email:
            raise ValidationError("请输入邮箱地址" if self.language.startswith("zh") else "Please enter an email address.")
        if User.objects.filter(email__iexact=email).exists():
            raise ValidationError("该邮箱已被注册" if self.language.startswith("zh") else "This email is already registered.")
        return email

    def save(self, commit: bool = True):
        user = super().save(commit=False)
        user.email = self.cleaned_data.get("email")
        if commit:
            user.save()
        return user


class ResendActivationForm(forms.Form):
    email = forms.EmailField(label="邮箱", required=True)

    def __init__(self, *args, language=None, **kwargs):
        self.language = (language or get_language() or "").lower()
        super().__init__(*args, **kwargs)
        label = "邮箱" if self.language.startswith("zh") else "Email"
        self.fields["email"].label = label
        self.fields["email"].widget.attrs.setdefault("placeholder", label)
        self.fields["email"].widget.attrs.setdefault("class", "form-control")
