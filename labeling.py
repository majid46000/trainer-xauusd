from __future__ import annotations
import numpy as np
import pandas as pd


def add_labels(
    df: pd.DataFrame,
    horizon: int = 3,
    threshold: float = 0.0005,           # الحد الأدنى للـ return (pipettes/نقاط)
    use_context: bool = True,             # تفعيل تعديل الـ labels بناءً على الاستراتيجيات
    fvg_boost: float = 1.4,               # مضاعف للـ future_return داخل FVG
    breakout_boost: float = 1.8,          # مضاعف للـ breakout قوي
    trend_min_strength: float = 1.0,      # متوسط trend_dir المطلوب على نافذة 5 شمعات
    macro_min_corr: float = -0.5,         # الحد الأدنى لارتباط سلبي مع DXY (اختياري)
    min_future_samples: int = 10          # الحد الأدنى لعدد الصفوف بعد الـ shift
) -> pd.DataFrame:
    """
    توليد labels متقدمة مع دعم الاستراتيجيات:
      - label =  1 → bullish (long signal)
      - label = -1 → bearish (short signal أو تجنب long)
      - label =  0 → neutral / flat

    Args:
        use_context: إذا True → يستخدم FVG, breakout, trend_dir, macro correlations
        fvg_boost / breakout_boost: مضاعف للـ future_return في المناطق القوية
        trend_min_strength: قوة الاتجاه المطلوبة (متوسط rolling)
        macro_min_corr: إذا كان الارتباط مع DXY أقل من هذا القيمة → زيادة ثقة الإشارة

    Returns:
        DataFrame مع أعمدة: future_return, label
    """
    df = df.copy()

    # ── الحساب الأساسي لـ future return ──
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["future_log_return"] = np.log(df["close"]).diff(horizon).shift(-horizon)

    # ── الـ label الأساسي (threshold-based) ──
    df["label"] = 0
    df.loc[df["future_return"] > threshold, "label"] = 1
    df.loc[df["future_return"] < -threshold, "label"] = -1

    if not use_context:
        df = df.dropna(subset=["future_return", "label"]).reset_index(drop=True)
        print_label_distribution(df)
        return df

    # ── تعديل الـ labels بناءً على الاستراتيجيات المتقدمة ──

    # 1. Fair Value Gap (FVG) → زيادة الثقة في الاتجاه داخل FVG
    if "bull_fvg" in df.columns and "bear_fvg" in df.columns:
        bull_fvg_mask = (df["bull_fvg"] == 1) & (df["future_return"] > 0)
        bear_fvg_mask = (df["bear_fvg"] == 1) & (df["future_return"] < 0)
        
        df.loc[bull_fvg_mask, "future_return"] *= fvg_boost
        df.loc[bear_fvg_mask, "future_return"] *= fvg_boost

    # 2. Breakout → تضخيم الإشارات بعد breakout قوي
    if "breakout_up" in df.columns and "breakout_down" in df.columns:
        up_break_mask = (df["breakout_up"] == 1) & (df["future_return"] > 0)
        down_break_mask = (df["breakout_down"] == 1) & (df["future_return"] < 0)
        
        df.loc[up_break_mask, "future_return"] *= breakout_boost
        df.loc[down_break_mask, "future_return"] *= breakout_boost

    # 3. Trend strength filter → إلغاء الإشارات المعاكسة للاتجاه المهيمن
    if "trend_dir" in df.columns:
        # حساب متوسط قوة الاتجاه على نافذة 5 شمعات
        trend_strength = df["trend_dir"].rolling(5, min_periods=3).mean()
        
        strong_bull = trend_strength >= trend_min_strength
        strong_bear = trend_strength <= -trend_min_strength
        
        # إلغاء الإشارات المعاكسة
        df.loc[~strong_bull & (df["label"] == 1), "label"] = 0
        df.loc[~strong_bear & (df["label"] == -1), "label"] = 0

    # 4. Macro correlation filter (اختياري - إذا كانت متوفرة)
    if "gold_dxy_corr" in df.columns:
        # الذهب يرتفع عادة عندما يضعف الدولار (correlation سلبي)
        inverse_corr_mask = df["gold_dxy_corr"] < macro_min_corr
        # تقليل ثقة الإشارات في حالة ارتباط ضعيف أو إيجابي
        df.loc[~inverse_corr_mask & (df["label"] != 0), "label"] *= 0.6

    # ── إعادة حساب الـ label النهائي بعد التعديلات ──
    df["label"] = np.select(
        [
            df["future_return"] > threshold,
            df["future_return"] < -threshold,
        ],
        [1, -1],
        default=0
    )

    # ── تنظيف NaN الناتجة عن shift(-horizon) ──
    df = df.dropna(subset=["future_return", "label"]).reset_index(drop=True)

    # طباعة توزيع الـ labels للتحقق
    print_label_distribution(df)

    return df


def print_label_distribution(df: pd.DataFrame) -> None:
    """طباعة توزيع الـ labels للتحقق من التوازن"""
    if "label" not in df.columns:
        print("Warning: 'label' column not found.")
        return
    
    dist = df["label"].value_counts(normalize=True).round(4) * 100
    print("\nLabel Distribution (percentage):")
    print(dist)
    print(f"Total samples after labeling: {len(df):,}")
    print("-" * 50)
