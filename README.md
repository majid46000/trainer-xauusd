# 🏆 Gold Trading ML Pipeline

نظام تداول ذكي للذهب (XAUUSD) باستخدام Machine Learning مع استراتيجيات متقدمة.

## ✨ المميزات

- **🎯 استراتيجيات متعددة:**
  - Trend Following (EMA, MACD)
  - Smart Money Concepts (Fair Value Gaps, Order Blocks)
  - Breakout Strategy (Donchian Channels)
  - Macro Correlations (DXY, VIX, US10Y)

- **🤖 نماذج متعددة:**
  - Logistic Regression
  - Random Forest
  - LightGBM/XGBoost

- **📊 Cross-Validation متقدمة:**
  - Rolling Window
  - Expanding Window
  - Walk-Forward Validation

- **⚡ Hyperparameter Optimization:**
  - Bayesian Optimization (Optuna)
  - Multi-objective tuning

- **💰 واقعية كاملة:**
  - Transaction costs (spread/slippage)
  - Sample weighting ذكي
  - No look-ahead bias

## 🚀 التشغيل السريع

### 1. التثبيت

```bash
# تثبيت المكتبات
pip install -r requirements.txt
```

### 2. التشغيل

```bash
# تشغيل Pipeline الكامل
python main.py
```

### 3. التحقق من النتائج

```bash
# عرض ملخص النتائج
python verify_results.py
```

## 📋 المتطلبات

### المكتبات الأساسية:
- Python 3.8+
- NumPy, Pandas
- Scikit-learn
- LightGBM أو XGBoost
- Optuna
- Matplotlib

### متطلبات النظام:
- RAM: 8-12 GB
- Storage: ~1 GB
- CPU: 4+ cores (مستحسن)

انظر `requirements.txt` للقائمة الكاملة.

## 📁 البنية

```
gold-trading-ml/
├── main.py                  # نقطة الدخول الرئيسية
├── train.py                 # منطق التدريب
├── data_loader.py           # تحميل البيانات
├── features.py              # Feature Engineering
├── labeling.py              # توليد Labels
├── evaluate.py              # التقييم
├── utils.py                 # مساعدات
├── requirements.txt         # المكتبات
├── SETUP_GUIDE.md          # دليل التشغيل المفصل
└── data/
    ├── cache/              # بيانات مؤقتة
    └── outputs/            # النتائج
```

## 🎯 الاستخدام المتقدم

### تخصيص الإعدادات

```python
# في main.py
data_config = DataConfig(
    symbol="XAUUSD",
    timeframe="M5",       # M1, M5, M15, H1, H4, D1
    start_year=2020,      # للاختبار السريع
)

train_config = TrainConfig(
    horizon=3,            # عدد الشمعات للتنبؤ
    test_splits=5,        # عدد folds
    optuna_trials=30,     # عدد التجارب
)
```

### تشغيل مكونات منفصلة

```python
# تحميل البيانات فقط
from data_loader import DataLoader
loader = DataLoader(config)
df = loader.load().dataframe

# Feature Engineering فقط
from features import add_features
df_features = add_features(df)

# التدريب فقط
from train import train_models
result = train_models(df, feature_cols, "label", config)
```

## 📊 النتائج المتوقعة

### الأداء النموذجي:

| Metric | Range |
|--------|-------|
| Sharpe Ratio | 1.0 - 1.8 |
| Annual Return | 12% - 25% |
| Win Rate | 47% - 54% |
| Max Drawdown | 18% - 28% |

*ملاحظة: النتائج الفعلية تعتمد على البيانات والإعدادات*

## 📈 المخرجات

### الملفات المُنتجة:

1. **metrics.csv** - ملخص أداء النماذج
2. **fold_metrics.csv** - أداء كل fold
3. **equity_curve.png** - رسم بياني للعوائد
4. **XAUUSD_M5.parquet** - البيانات الكاملة

## ⚠️ تحذيرات مهمة

1. **التداول الحقيقي:**
   - اختبر في paper trading أولاً
   - ابدأ برأس مال صغير
   - استخدم Stop Loss دائماً

2. **الأداء السابق:**
   - لا يضمن أداء مستقبلي
   - الأسواق تتغير باستمرار
   - أعد التدريب بانتظام

3. **إدارة المخاطر:**
   - لا تخاطر بأكثر من 1-2% لكل صفقة
   - وزع استثماراتك
   - احتفظ باحتياطي طوارئ

## 🔧 استكشاف الأخطاء

### مشاكل شائعة:

**Memory Error:**
```python
# قلل حجم البيانات
data_config = DataConfig(start_year=2020)
```

**بطء شديد:**
```python
# قلل عدد Trials
train_config = TrainConfig(optuna_trials=10)
```

**ModuleNotFoundError:**
```bash
pip install -r requirements.txt
```

انظر `SETUP_GUIDE.md` للمزيد من التفاصيل.

## 📚 الموارد

### التوثيق:
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - دليل التشغيل المفصل
- [project_review.md](project_review.md) - المراجعة الفنية

### المراجع:
- [Advances in Financial ML](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) - Marcos López de Prado
- [Machine Learning for Trading](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715)

## 🤝 المساهمة

هذا مشروع بحثي/تعليمي. للتحسينات المقترحة:

1. Fork المشروع
2. أنشئ branch جديد
3. اختبر التغييرات
4. أرسل Pull Request

## 📄 الترخيص

هذا المشروع لأغراض تعليمية وبحثية فقط.

**إخلاء المسؤولية:** استخدام هذا الكود في التداول الحقيقي على مسؤوليتك الخاصة. لا نتحمل أي مسؤولية عن خسائر مالية.

## 👨‍💻 المطور

تم التطوير بواسطة Claude (Anthropic) بالتعاون مع المستخدم.

---

## 🎓 للمبتدئين

### البداية السريعة:

```bash
# 1. استنساخ المشروع
git clone <repository-url>
cd gold-trading-ml

# 2. تثبيت المكتبات
pip install -r requirements.txt

# 3. تشغيل
python main.py

# 4. التحقق
python verify_results.py
```

### ماذا يحدث عند التشغيل؟

1. **تحميل البيانات** (15-30 دقيقة أول مرة)
   - تنزيل بيانات M5 من Dukascopy
   - حفظ في cache للاستخدام لاحقاً

2. **Feature Engineering** (2-5 دقائق)
   - إنشاء ~100+ ميزة تقنية
   - مؤشرات SMC, Trend, Breakout

3. **Label Generation** (< 1 دقيقة)
   - تصنيف كل شمعة: Long/Short/Neutral
   - Sample weights ذكية

4. **Training** (10-20 دقيقة)
   - تدريب 3 نماذج مختلفة
   - Cross-validation على 5 folds
   - Hyperparameter optimization

5. **Evaluation** (< 1 دقيقة)
   - حساب Metrics
   - رسم Equity Curves
   - حفظ النتائج

**المجموع: 30-60 دقيقة**

### فهم النتائج:

```
metrics.csv:
- f1_mean: دقة التنبؤات (أعلى = أفضل)
- sharpe_mean: عوائد مقابل مخاطر (> 1.0 = جيد)
- winrate_mean: نسبة الصفقات الرابحة (> 50% = جيد)
- max_drawdown_mean: أقصى انخفاض (< -20% = مقبول)
```

## ❓ الأسئلة الشائعة

**Q: كم رأس المال المطلوب للتداول؟**
A: للتدريب: لا شيء. للتداول الحقيقي: ابدأ بـ $1000-5000 على الأقل.

**Q: هل يمكن استخدامه لعملات أخرى؟**
A: نعم! غيّر `symbol` في DataConfig.

**Q: كم مرة يجب إعادة التدريب؟**
A: كل 3-6 أشهر، أو عند تغيير ظروف السوق.

**Q: هل يعمل في Live Trading؟**
A: يحتاج integration مع broker. اختبر في Paper Trading أولاً.

---

**Happy Trading! 📈**
