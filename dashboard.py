import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dashboard Analisis E-Commerce Olist",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS KUSTOM ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .main { background-color: #f1f5f9; }

    [data-testid="metric-container"] {
        background: white;
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 5px solid #2563eb;
    }
    [data-testid="metric-container"] label {
        color: #64748b;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #0f172a;
        font-size: 24px;
        font-weight: 800;
    }

    .judul-seksi {
        font-size: 17px;
        font-weight: 800;
        color: #0f172a;
        padding-bottom: 7px;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 4px;
    }
    .sub-seksi {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 14px;
        font-style: italic;
    }

    .kotak-insight {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        font-size: 13.5px;
        color: #1e3a8a;
        margin-top: 10px;
        line-height: 1.6;
    }

    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] hr { border-color: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ── PALET WARNA ────────────────────────────────────────────────────────────
C_BIRU   = '#2563eb'
C_MERAH  = '#dc2626'
C_HIJAU  = '#16a34a'
C_KUNING = '#d97706'
C_UNGU   = '#7c3aed'
C_ABU    = '#94a3b8'
C_BG     = '#f8fafc'

WARNA_SEGMEN = {
    'Juara':     '#16a34a',
    'Setia':     '#2563eb',
    'Potensial': '#d97706',
    'Berisiko':  '#ea580c',
    'Hilang':    '#dc2626',
    'Baru':      '#7c3aed',
}

def rapikan_ax(ax, bg=C_BG):
    ax.set_facecolor(bg)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(colors='#475569', labelsize=9)
    ax.xaxis.label.set_color('#334155')
    ax.yaxis.label.set_color('#334155')
    ax.title.set_color('#0f172a')
    ax.grid(axis='y', color='#e2e8f0', linewidth=0.6, alpha=0.8)


# ── MUAT DATA ──────────────────────────────────────────────────────────────
@st.cache_data
def muat_data():
    df_pesanan      = pd.read_csv("orders_dataset.csv")
    df_item         = pd.read_csv("order_items_dataset.csv")
    df_pembayaran   = pd.read_csv("order_payments_dataset.csv")
    df_ulasan       = pd.read_csv("order_reviews_dataset.csv")
    df_pelanggan    = pd.read_csv("customers_dataset.csv")
    df_produk       = pd.read_csv("products_dataset.csv")
    df_penjual      = pd.read_csv("sellers_dataset.csv")
    df_kategori     = pd.read_csv("product_category_name_translation.csv")

    # Konversi timestamp
    kolom_waktu = [
        'order_purchase_timestamp', 'order_approved_at',
        'order_delivered_carrier_date', 'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for k in kolom_waktu:
        df_pesanan[k] = pd.to_datetime(df_pesanan[k], errors='coerce')

    # Bersihkan produk
    df_produk['product_category_name'].fillna('outros', inplace=True)
    df_produk['product_weight_g'].fillna(df_produk['product_weight_g'].median(), inplace=True)

    # Perbaiki cicilan 0
    df_pembayaran.loc[df_pembayaran['payment_installments'] == 0, 'payment_installments'] = 1

    # Filter hanya pesanan terkirim
    df_terkirim = df_pesanan[df_pesanan['order_status'] == 'delivered'].copy()
    df_terkirim = df_terkirim.dropna(
        subset=['order_delivered_customer_date', 'order_estimated_delivery_date']
    )

    # Fitur turunan pengiriman
    df_terkirim['durasi_pengiriman_hari'] = (
        df_terkirim['order_delivered_customer_date'] -
        df_terkirim['order_purchase_timestamp']
    ).dt.days

    df_terkirim['selisih_estimasi_hari'] = (
        df_terkirim['order_delivered_customer_date'] -
        df_terkirim['order_estimated_delivery_date']
    ).dt.days

    df_terkirim['status_pengiriman'] = df_terkirim['selisih_estimasi_hari'].apply(
        lambda x: 'Terlambat' if x > 0 else 'Tepat Waktu'
    )
    df_terkirim['bulan_pesanan'] = df_terkirim['order_purchase_timestamp'].dt.strftime('%Y-%m')
    df_terkirim['tahun_pesanan'] = df_terkirim['order_purchase_timestamp'].dt.year

    # Gabungkan tabel
    df_prod_kat = df_produk.merge(df_kategori, on='product_category_name', how='left')
    df_item = df_item.merge(
        df_prod_kat[['product_id', 'product_category_name_english']],
        on='product_id', how='left'
    )

    df = df_terkirim.merge(df_item, on='order_id', how='inner')
    df = df.merge(
        df_ulasan[['order_id', 'review_score']].drop_duplicates('order_id'),
        on='order_id', how='left'
    )
    df = df.merge(
        df_pelanggan[['customer_id', 'customer_unique_id', 'customer_state']],
        on='customer_id', how='left'
    )
    df = df.merge(
        df_pembayaran[['order_id', 'payment_type', 'payment_installments', 'payment_value']]
        .drop_duplicates('order_id'),
        on='order_id', how='left'
    )

    df['nilai_total'] = df['price'] + df['freight_value']

    # Kelompok cicilan
    df['kelompok_cicilan'] = pd.cut(
        df['payment_installments'],
        bins=[0, 1, 3, 6, 12, 100],
        labels=['1x', '2–3x', '4–6x', '7–12x', '>12x']
    )

    return df, df_penjual, df_item


df_semua, df_penjual_raw, df_item_raw = muat_data()


# ── SIDEBAR FILTER ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 Olist Analytics")
    st.markdown("**Dashboard E-Commerce Brasil**")
    st.markdown("---")
    st.markdown("### 🔧 Filter Data")

    daftar_tahun = sorted(df_semua['tahun_pesanan'].dropna().unique().astype(int).tolist())
    pilih_tahun  = st.multiselect("Tahun Pesanan", daftar_tahun, default=daftar_tahun)

    daftar_status = ['Tepat Waktu', 'Terlambat']
    pilih_status  = st.multiselect("Status Pengiriman", daftar_status, default=daftar_status)

    daftar_bayar = sorted(df_semua['payment_type'].dropna().unique().tolist())
    pilih_bayar  = st.multiselect("Metode Pembayaran", daftar_bayar, default=daftar_bayar)

    st.markdown("---")
    st.markdown("**Dataset:** Brazilian E-Commerce  \nPublic Dataset by Olist")
    st.markdown("**Periode:** Jan 2017 – Ags 2018")
    st.markdown("---")
    st.markdown("Abdul Hamid Amin")

# Terapkan filter
if pilih_tahun and pilih_status and pilih_bayar:
    df = df_semua[
        df_semua['tahun_pesanan'].isin(pilih_tahun) &
        df_semua['status_pengiriman'].isin(pilih_status) &
        df_semua['payment_type'].isin(pilih_bayar)
    ].copy()
else:
    df = df_semua.copy()


# ── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("# 📦 Dashboard Analisis E-Commerce Olist")
st.markdown(
    "Analisis mendalam tentang **ketepatan pengiriman**, **metode pembayaran**, "
    "dan **segmentasi pelanggan** platform e-commerce Olist Brasil."
)
st.markdown("---")


# ── KPI UTAMA ──────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
pct_tepat = (df['status_pengiriman'] == 'Tepat Waktu').mean() * 100
k1.metric("📦 Total Pesanan",        f"{df['order_id'].nunique():,}")
k2.metric("💰 Total Pendapatan",     f"R$ {df['nilai_total'].sum()/1e6:.2f}M")
k3.metric("⭐ Rata-rata Ulasan",     f"{df['review_score'].mean():.2f} / 5")
k4.metric("🚚 Tepat Waktu",         f"{pct_tepat:.1f}%")
k5.metric("👤 Pelanggan Unik",      f"{df['customer_unique_id'].nunique():,}")

st.markdown("---")


# ── VIZ 1: KETEPATAN PENGIRIMAN ────────────────────────────────────────────
st.markdown('<p class="judul-seksi">🚚 Pertanyaan 1 — Analisis Ketepatan Waktu Pengiriman</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-seksi">Bagaimana tingkat ketepatan pengiriman Olist dan dampaknya terhadap kepuasan pelanggan?</p>', unsafe_allow_html=True)

kol_v1a, kol_v1b = st.columns([3, 1])

with kol_v1a:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor(C_BG)

    # Panel 1: Pie proporsi
    status_count = df['status_pengiriman'].value_counts()
    warna_pie = [C_HIJAU if s == 'Tepat Waktu' else C_MERAH for s in status_count.index]
    wedges, texts, autotexts = axes[0].pie(
        status_count.values, labels=status_count.index,
        autopct='%1.1f%%', colors=warna_pie, startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2.5),
        textprops=dict(fontsize=10)
    )
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(11)
    axes[0].set_title('Proporsi Status\nPengiriman', fontsize=11, fontweight='bold')
    axes[0].set_facecolor(C_BG)

    # Panel 2: Histogram durasi
    dur = df['durasi_pengiriman_hari'].dropna()
    dur = dur[(dur >= 0) & (dur <= 60)]
    axes[1].hist(dur, bins=28, color=C_BIRU, edgecolor='white', linewidth=0.5, alpha=0.85)
    axes[1].axvline(dur.mean(), color=C_MERAH, linestyle='--', linewidth=1.5,
                    label=f'Rata-rata: {dur.mean():.1f} hr')
    axes[1].axvline(dur.median(), color=C_KUNING, linestyle='--', linewidth=1.5,
                    label=f'Median: {dur.median():.1f} hr')
    axes[1].set_xlabel('Durasi (Hari)')
    axes[1].set_ylabel('Frekuensi')
    axes[1].set_title('Distribusi Durasi\nPengiriman', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=8)
    rapikan_ax(axes[1])

    # Panel 3: Skor ulasan per status
    ulasan_stat = df.groupby('status_pengiriman')['review_score'].mean().sort_values(ascending=False)
    w_bar = [C_HIJAU if s == 'Tepat Waktu' else C_MERAH for s in ulasan_stat.index]
    batang = axes[2].bar(ulasan_stat.index, ulasan_stat.values, color=w_bar, width=0.4, alpha=0.88)
    for b, v in zip(batang, ulasan_stat.values):
        axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 0.03,
                     f'{v:.2f} ⭐', ha='center', fontsize=11, fontweight='bold')
    axes[2].set_ylim(0, 5.6)
    axes[2].set_ylabel('Rata-rata Skor Ulasan')
    axes[2].set_title('Skor Ulasan: Tepat Waktu\nvs Terlambat', fontsize=11, fontweight='bold')
    rapikan_ax(axes[2])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with kol_v1b:
    st.markdown("#### Statistik Pengiriman")
    dur_clean = df['durasi_pengiriman_hari'].dropna()
    st.metric("Rata-rata Durasi",    f"{dur_clean.mean():.1f} hari")
    st.metric("Durasi Terpendek",    f"{dur_clean.min():.0f} hari")
    st.metric("Durasi Terpanjang",   f"{dur_clean.max():.0f} hari")
    st.metric("Median Durasi",       f"{dur_clean.median():.1f} hari")
    st.markdown("---")

    # Top 5 negara bagian terlambat
    st.markdown("#### 🏴 Terlambat Terbanyak")
    df_terlambat = df[df['status_pengiriman'] == 'Terlambat']
    top_lambat = df_terlambat['customer_state'].value_counts().head(5)
    for negbag, jml in top_lambat.items():
        st.markdown(f"**{negbag}** — {jml:,} pesanan")

# Tren bulanan keterlambatan
st.markdown("##### Tren Persentase Keterlambatan per Bulan")
tren_bulanan = (
    df.groupby('bulan_pesanan')['status_pengiriman']
    .apply(lambda x: (x == 'Terlambat').mean() * 100)
    .reset_index()
    .sort_values('bulan_pesanan')
)
tren_bulanan.columns = ['Bulan', 'Pct_Terlambat']

fig_tren, ax_tren = plt.subplots(figsize=(14, 3.5))
fig_tren.patch.set_facecolor(C_BG)
ax_tren.fill_between(range(len(tren_bulanan)), tren_bulanan['Pct_Terlambat'],
                     alpha=0.3, color=C_MERAH)
ax_tren.plot(range(len(tren_bulanan)), tren_bulanan['Pct_Terlambat'],
             color=C_MERAH, linewidth=2, marker='o', markersize=5)
ax_tren.axhline(tren_bulanan['Pct_Terlambat'].mean(), color=C_ABU,
                linestyle='--', linewidth=1.2,
                label=f"Rata-rata: {tren_bulanan['Pct_Terlambat'].mean():.1f}%")
ax_tren.set_xticks(range(len(tren_bulanan)))
ax_tren.set_xticklabels(tren_bulanan['Bulan'], rotation=45, ha='right', fontsize=8)
ax_tren.set_ylabel('% Pesanan Terlambat')
ax_tren.set_title('Persentase Keterlambatan Pengiriman per Bulan', fontsize=12, fontweight='bold')
ax_tren.legend(fontsize=9)
rapikan_ax(ax_tren)
plt.tight_layout()
st.pyplot(fig_tren)
plt.close()

pct_tp = (df['status_pengiriman'] == 'Tepat Waktu').mean() * 100
rata_dur = df['durasi_pengiriman_hari'].mean()
ulasan_tepat = df[df['status_pengiriman'] == 'Tepat Waktu']['review_score'].mean()
ulasan_lambat = df[df['status_pengiriman'] == 'Terlambat']['review_score'].mean()
st.markdown(
    f'<div class="kotak-insight">💡 <b>Insight:</b> Sebesar <b>{pct_tp:.1f}%</b> pesanan berhasil dikirim '
    f'tepat waktu dengan rata-rata durasi <b>{rata_dur:.1f} hari</b>. Pesanan tepat waktu mendapat skor '
    f'ulasan rata-rata <b>{ulasan_tepat:.2f}</b>, sedangkan pesanan terlambat hanya <b>{ulasan_lambat:.2f}</b> — '
    f'selisih yang signifikan dan membuktikan dampak langsung keterlambatan terhadap kepuasan pelanggan.</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ── VIZ 2: METODE PEMBAYARAN ───────────────────────────────────────────────
st.markdown('<p class="judul-seksi">💳 Pertanyaan 2 — Analisis Metode Pembayaran & Pola Cicilan</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-seksi">Metode apa yang paling populer dan bagaimana cicilan mempengaruhi nilai transaksi?</p>', unsafe_allow_html=True)

ringkasan_bayar = (
    df.groupby('payment_type')
    .agg(
        jumlah_transaksi=('order_id', 'nunique'),
        rata_nilai=('payment_value', 'mean'),
        rata_ulasan=('review_score', 'mean'),
        rata_cicilan=('payment_installments', 'mean')
    )
    .reset_index()
    .sort_values('jumlah_transaksi', ascending=False)
)

kol_v2a, kol_v2b = st.columns(2)

with kol_v2a:
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
    fig2.patch.set_facecolor(C_BG)

    # Bar horizontal jumlah transaksi
    rb_sort = ringkasan_bayar.sort_values('jumlah_transaksi')
    warna_bayar = [C_BIRU, C_UNGU, C_KUNING, C_HIJAU][:len(rb_sort)]
    batang_b = axes2[0].barh(rb_sort['payment_type'], rb_sort['jumlah_transaksi'],
                              color=warna_bayar, alpha=0.88)
    for b, v in zip(batang_b, rb_sort['jumlah_transaksi']):
        axes2[0].text(v + 5, b.get_y() + b.get_height()/2,
                      f'{v:,}', va='center', fontsize=9, fontweight='bold')
    axes2[0].set_xlabel('Jumlah Transaksi')
    axes2[0].set_title('Jumlah Transaksi\nper Metode Pembayaran', fontsize=11, fontweight='bold')
    rapikan_ax(axes2[0])
    axes2[0].grid(axis='x', color='#e2e8f0', linewidth=0.6)
    axes2[0].grid(axis='y', visible=False)

    # Rata-rata nilai per metode
    rb_nilai = ringkasan_bayar.sort_values('rata_nilai')
    warna_nilai = [C_BIRU, C_UNGU, C_KUNING, C_HIJAU][:len(rb_nilai)]
    batang_n = axes2[1].barh(rb_nilai['payment_type'], rb_nilai['rata_nilai'],
                              color=warna_nilai, alpha=0.88)
    for b, v in zip(batang_n, rb_nilai['rata_nilai']):
        axes2[1].text(v + 1, b.get_y() + b.get_height()/2,
                      f'R${v:,.0f}', va='center', fontsize=9, fontweight='bold')
    axes2[1].set_xlabel('Rata-rata Nilai (R$)')
    axes2[1].set_title('Rata-rata Nilai Transaksi\nper Metode', fontsize=11, fontweight='bold')
    rapikan_ax(axes2[1])
    axes2[1].grid(axis='x', color='#e2e8f0', linewidth=0.6)
    axes2[1].grid(axis='y', visible=False)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

with kol_v2b:
    # Bar rata-rata nilai per kelompok cicilan
    rata_kelompok = (
        df.groupby('kelompok_cicilan', observed=True)['payment_value']
        .agg(['mean', 'count'])
        .reset_index()
    )
    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    fig3.patch.set_facecolor(C_BG)
    warna_klp = [C_HIJAU, C_BIRU, C_KUNING, C_MERAH, C_UNGU][:len(rata_kelompok)]
    batang_k = ax3.bar(rata_kelompok['kelompok_cicilan'].astype(str),
                       rata_kelompok['mean'], color=warna_klp, alpha=0.88)
    for b, v in zip(batang_k, rata_kelompok['mean']):
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
                 f'R${v:,.0f}', ha='center', fontsize=8, fontweight='bold')
    ax3.set_xlabel('Kelompok Cicilan')
    ax3.set_ylabel('Rata-rata Nilai Transaksi (R$)')
    ax3.set_title('Rata-rata Nilai Transaksi\nper Kelompok Cicilan', fontsize=11, fontweight='bold')
    rapikan_ax(ax3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# Tabel ringkasan pembayaran
st.markdown("##### Ringkasan Statistik per Metode Pembayaran")
tbl_bayar = ringkasan_bayar.copy()
tbl_bayar.columns = ['Metode', 'Jml Transaksi', 'Rata Nilai (R$)', 'Rata Ulasan', 'Rata Cicilan']
tbl_bayar['Rata Nilai (R$)'] = tbl_bayar['Rata Nilai (R$)'].round(2)
tbl_bayar['Rata Ulasan']     = tbl_bayar['Rata Ulasan'].round(2)
tbl_bayar['Rata Cicilan']    = tbl_bayar['Rata Cicilan'].round(1)
st.dataframe(tbl_bayar.set_index('Metode'), use_container_width=True)

cc = df[['payment_installments', 'payment_value']].corr().iloc[0, 1]
metode_dominan = ringkasan_bayar.iloc[0]['payment_type']
pct_dominan = ringkasan_bayar.iloc[0]['jumlah_transaksi'] / ringkasan_bayar['jumlah_transaksi'].sum() * 100
st.markdown(
    f'<div class="kotak-insight">💡 <b>Insight:</b> <b>{metode_dominan.replace("_", " ").title()}</b> '
    f'mendominasi dengan <b>{pct_dominan:.1f}%</b> dari total transaksi. Korelasi antara jumlah cicilan '
    f'dan nilai pembayaran sebesar <b>{cc:.3f}</b> — menunjukkan hubungan positif yang jelas: '
    f'semakin banyak cicilan dipilih, semakin besar nilai transaksinya.</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ── RFM ANALYSIS ──────────────────────────────────────────────────────────
st.markdown('<p class="judul-seksi">🎯 Analisis Lanjutan — Segmentasi Pelanggan (RFM)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-seksi">Mengelompokkan pelanggan berdasarkan Recency, Frequency, dan Monetary untuk strategi retensi yang tepat sasaran.</p>', unsafe_allow_html=True)


@st.cache_data
def hitung_rfm(df_input):
    tgl_ref = df_input['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = (
        df_input.groupby('customer_unique_id')
        .agg(
            Recency=('order_purchase_timestamp', lambda x: (tgl_ref - x.max()).days),
            Frequency=('order_id', 'nunique'),
            Monetary=('nilai_total', 'sum')
        )
        .reset_index()
    )
    rfm['skor_R'] = pd.qcut(rfm['Recency'], q=5, labels=[5,4,3,2,1], duplicates='drop').astype(int)
    rfm['skor_F'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
    rfm['skor_M'] = pd.qcut(rfm['Monetary'], q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)

    def segmen(r):
        rc, f, m = r['skor_R'], r['skor_F'], r['skor_M']
        if rc >= 4 and f >= 4 and m >= 4: return 'Juara'
        elif rc >= 4 and f >= 3:           return 'Setia'
        elif rc >= 3 and f >= 2 and m >= 3: return 'Potensial'
        elif rc <= 2 and f >= 3:           return 'Berisiko'
        elif rc <= 2 and f <= 1 and m <= 2: return 'Hilang'
        else:                              return 'Baru'

    rfm['segmen'] = rfm.apply(segmen, axis=1)
    return rfm


df_rfm = hitung_rfm(df_semua)

kol_r1, kol_r2, kol_r3 = st.columns(3)

with kol_r1:
    jml_seg = df_rfm['segmen'].value_counts()
    fig_r1, ax_r1 = plt.subplots(figsize=(5.5, 4))
    fig_r1.patch.set_facecolor(C_BG)
    w_s = [WARNA_SEGMEN.get(s, C_BIRU) for s in jml_seg.index]
    b_s = ax_r1.bar(jml_seg.index, jml_seg.values, color=w_s, alpha=0.88)
    for b, v in zip(b_s, jml_seg.values):
        ax_r1.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                   f'{v:,}', ha='center', fontsize=8, fontweight='bold')
    ax_r1.set_ylabel('Jumlah Pelanggan')
    ax_r1.set_title('Distribusi Segmen RFM', fontsize=11, fontweight='bold')
    ax_r1.tick_params(axis='x', rotation=20)
    rapikan_ax(ax_r1)
    plt.tight_layout()
    st.pyplot(fig_r1)
    plt.close()

with kol_r2:
    avg_m = df_rfm.groupby('segmen')['Monetary'].mean().sort_values()
    fig_r2, ax_r2 = plt.subplots(figsize=(5.5, 4))
    fig_r2.patch.set_facecolor(C_BG)
    w_m = [WARNA_SEGMEN.get(s, C_BIRU) for s in avg_m.index]
    ax_r2.barh(avg_m.index, avg_m.values, color=w_m, alpha=0.88)
    for i, v in enumerate(avg_m.values):
        ax_r2.text(v + 1, i, f'R${v:,.0f}', va='center', fontsize=8)
    ax_r2.set_xlabel('Rata-rata Belanja (R$)')
    ax_r2.set_title('Rata-rata Belanja per Segmen', fontsize=11, fontweight='bold')
    rapikan_ax(ax_r2)
    ax_r2.grid(axis='x', color='#e2e8f0', linewidth=0.6)
    ax_r2.grid(axis='y', visible=False)
    plt.tight_layout()
    st.pyplot(fig_r2)
    plt.close()

with kol_r3:
    fig_r3, ax_r3 = plt.subplots(figsize=(5.5, 4))
    fig_r3.patch.set_facecolor(C_BG)
    for seg, warna in WARNA_SEGMEN.items():
        sub = df_rfm[df_rfm['segmen'] == seg]
        if len(sub) > 0:
            ax_r3.scatter(sub['Recency'], sub['Monetary'],
                          c=warna, alpha=0.35, s=12, label=seg)
    ax_r3.set_xlabel('Recency (Hari)')
    ax_r3.set_ylabel('Total Belanja (R$)')
    ax_r3.set_title('Peta Recency vs Monetary', fontsize=11, fontweight='bold')
    ax_r3.legend(fontsize=7, framealpha=0.8)
    rapikan_ax(ax_r3)
    ax_r3.grid(color='#e2e8f0', linewidth=0.6)
    plt.tight_layout()
    st.pyplot(fig_r3)
    plt.close()

# Tabel ringkasan RFM
rfm_ringkas = (
    df_rfm.groupby('segmen')
    .agg(
        Jumlah_Pelanggan=('customer_unique_id', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean')
    )
    .reset_index()
    .sort_values('Avg_Monetary', ascending=False)
    .rename(columns={'segmen': 'Segmen'})
)
rfm_ringkas['Avg_Recency']   = rfm_ringkas['Avg_Recency'].round(0).astype(int).astype(str) + ' hari'
rfm_ringkas['Avg_Frequency'] = rfm_ringkas['Avg_Frequency'].round(1)
rfm_ringkas['Avg_Monetary']  = rfm_ringkas['Avg_Monetary'].apply(lambda x: f'R$ {x:,.2f}')
st.dataframe(rfm_ringkas.set_index('Segmen'), use_container_width=True)

st.markdown("---")


# ── GEOSPATIAL ─────────────────────────────────────────────────────────────
st.markdown('<p class="judul-seksi">🗺️ Analisis Lanjutan — Distribusi Geografis Penjual</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-seksi">Persebaran penjual dan volume pesanan per negara bagian di Brasil, serta efisiensi penjual per wilayah.</p>', unsafe_allow_html=True)

koordinat = {
    'SP': (-23.5, -46.6), 'RJ': (-22.9, -43.2), 'MG': (-19.9, -43.9),
    'RS': (-30.0, -51.2), 'PR': (-25.4, -49.3), 'SC': (-27.6, -48.5),
    'BA': (-12.9, -38.5), 'GO': (-16.7, -49.3), 'PE': (-8.1, -34.9),
    'CE': (-3.7, -38.5),  'ES': (-19.2, -40.3), 'MT': (-12.6, -55.9),
    'MS': (-20.4, -54.6), 'PB': (-7.2, -36.8),  'RN': (-5.8, -36.5),
    'AL': (-9.7, -36.6),  'MA': (-5.4, -45.4),  'PI': (-8.0, -43.0),
    'PA': (-3.4, -52.0),  'AM': (-4.4, -63.6),  'DF': (-15.8, -47.9),
    'RO': (-11.5, -63.6), 'TO': (-10.2, -48.3)
}


@st.cache_data
def hitung_geo(df_penjual_raw, df_item_raw):
    df_pj_item = df_item_raw.merge(df_penjual_raw, on='seller_id', how='left')
    geo = (
        df_pj_item.groupby('seller_state')
        .agg(
            jumlah_penjual=('seller_id', 'nunique'),
            jumlah_item=('order_id', 'count'),
        )
        .reset_index()
    )
    geo['pesanan_per_penjual'] = (geo['jumlah_item'] / geo['jumlah_penjual']).round(1)
    geo['lat'] = geo['seller_state'].map(lambda x: koordinat.get(x, (-15, -50))[0])
    geo['lon'] = geo['seller_state'].map(lambda x: koordinat.get(x, (-15, -50))[1])
    return geo.sort_values('jumlah_item', ascending=False)


geo_penjual = hitung_geo(df_penjual_raw, df_item_raw)

kol_g1, kol_g2 = st.columns([3, 2])

with kol_g1:
    ukuran_buble = geo_penjual['jumlah_penjual'] / geo_penjual['jumlah_penjual'].max() * 1800
    fig_g1, ax_g1 = plt.subplots(figsize=(8, 6))
    fig_g1.patch.set_facecolor('#0f172a')
    ax_g1.set_facecolor('#0f172a')

    sc_g = ax_g1.scatter(
        geo_penjual['lon'], geo_penjual['lat'],
        s=ukuran_buble,
        c=geo_penjual['jumlah_penjual'],
        cmap='YlOrRd', alpha=0.85,
        edgecolors='white', linewidth=0.5
    )
    cb_g = plt.colorbar(sc_g, ax=ax_g1, shrink=0.65)
    cb_g.set_label('Jumlah Penjual', color='white', fontsize=9)
    cb_g.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb_g.ax.axes, 'yticklabels'), color='white')

    for _, brs in geo_penjual.iterrows():
        ax_g1.text(brs['lon'], brs['lat'], brs['seller_state'],
                   ha='center', va='center', fontsize=7,
                   color='white', fontweight='bold')

    ax_g1.set_xlim(-75, -30)
    ax_g1.set_ylim(-35, 6)
    ax_g1.set_title('Peta Sebaran Penjual per Negara Bagian Brasil',
                    fontsize=12, fontweight='bold', color='white', pad=12)
    ax_g1.set_xlabel('Longitude', color='#94a3b8', fontsize=9)
    ax_g1.set_ylabel('Latitude',  color='#94a3b8', fontsize=9)
    ax_g1.tick_params(colors='#64748b')
    ax_g1.grid(color='#1e3a5f', linewidth=0.4, alpha=0.5)
    for spine in ax_g1.spines.values():
        spine.set_edgecolor('#1e3a5f')

    plt.tight_layout()
    st.pyplot(fig_g1)
    plt.close()

with kol_g2:
    top10_efisien = geo_penjual.sort_values('pesanan_per_penjual', ascending=False).head(10)
    fig_g2, axes_g2 = plt.subplots(2, 1, figsize=(5.5, 6))
    fig_g2.patch.set_facecolor(C_BG)

    # Jumlah penjual
    warna_g = plt.cm.YlOrRd(np.linspace(0.4, 0.9, 10))[::-1]
    top10_jml = geo_penjual.head(10)
    axes_g2[0].barh(top10_jml['seller_state'], top10_jml['jumlah_penjual'],
                    color=warna_g, alpha=0.9)
    axes_g2[0].invert_yaxis()
    axes_g2[0].set_title('Top 10 — Jumlah Penjual', fontsize=10, fontweight='bold')
    for i, v in enumerate(top10_jml['jumlah_penjual']):
        axes_g2[0].text(v + 1, i, f'{v:,}', va='center', fontsize=8)
    rapikan_ax(axes_g2[0])
    axes_g2[0].grid(axis='x', color='#e2e8f0', linewidth=0.6)
    axes_g2[0].grid(axis='y', visible=False)

    # Efisiensi
    axes_g2[1].barh(top10_efisien['seller_state'], top10_efisien['pesanan_per_penjual'],
                    color=plt.cm.Blues(np.linspace(0.4, 0.9, 10))[::-1], alpha=0.9)
    axes_g2[1].invert_yaxis()
    axes_g2[1].set_title('Top 10 — Efisiensi (Pesanan/Penjual)', fontsize=10, fontweight='bold')
    for i, v in enumerate(top10_efisien['pesanan_per_penjual']):
        axes_g2[1].text(v + 0.1, i, f'{v:.1f}x', va='center', fontsize=8)
    rapikan_ax(axes_g2[1])
    axes_g2[1].grid(axis='x', color='#e2e8f0', linewidth=0.6)
    axes_g2[1].grid(axis='y', visible=False)

    plt.tight_layout()
    st.pyplot(fig_g2)
    plt.close()

st.markdown(
    '<div class="kotak-insight">💡 <b>Insight:</b> Penjual sangat terkonsentrasi di negara bagian '
    '<b>SP (São Paulo)</b>. Namun beberapa negara bagian lain menunjukkan efisiensi tinggi '
    '(pesanan per penjual) yang mengindikasikan peluang rekrutmen penjual baru di wilayah tersebut '
    'untuk mengurangi ketergantungan pada satu wilayah dan memperluas jangkauan pasar.</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ── BINNING PELANGGAN ──────────────────────────────────────────────────────
st.markdown('<p class="judul-seksi">📊 Analisis Lanjutan — Segmentasi Nilai Belanja (Binning)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-seksi">Pengelompokan pelanggan berdasarkan total nilai belanja menggunakan metode binning kuartil.</p>', unsafe_allow_html=True)


@st.cache_data
def hitung_binning(df_input):
    total_blj = (
        df_input.groupby('customer_unique_id')
        .agg(
            total_belanja=('nilai_total', 'sum'),
            jumlah_pesanan=('order_id', 'nunique'),
            rata_ulasan=('review_score', 'mean')
        )
        .reset_index()
    )
    q25 = total_blj['total_belanja'].quantile(0.25)
    q50 = total_blj['total_belanja'].quantile(0.50)
    q75 = total_blj['total_belanja'].quantile(0.75)

    def kategori(v):
        if v <= q25:   return 'Hemat (≤Q1)'
        elif v <= q50: return 'Menengah (Q1–Q2)'
        elif v <= q75: return 'Aktif (Q2–Q3)'
        else:          return 'Premium (>Q3)'

    total_blj['kelompok'] = total_blj['total_belanja'].apply(kategori)
    return total_blj


df_binning = hitung_binning(df_semua)
URUTAN = ['Hemat (≤Q1)', 'Menengah (Q1–Q2)', 'Aktif (Q2–Q3)', 'Premium (>Q3)']
WARNA_KLPK = {'Hemat (≤Q1)': C_ABU, 'Menengah (Q1–Q2)': C_BIRU,
               'Aktif (Q2–Q3)': C_KUNING, 'Premium (>Q3)': C_HIJAU}

ringkas_bin = (
    df_binning.groupby('kelompok')
    .agg(
        jumlah_pelanggan=('customer_unique_id', 'count'),
        avg_belanja=('total_belanja', 'mean'),
        avg_ulasan=('rata_ulasan', 'mean'),
        avg_pesanan=('jumlah_pesanan', 'mean')
    )
    .reindex(URUTAN)
    .reset_index()
)
w_k = [WARNA_KLPK[k] for k in ringkas_bin['kelompok']]

fig_bin, axes_bin = plt.subplots(1, 3, figsize=(14, 4))
fig_bin.patch.set_facecolor(C_BG)

# Panel 1
b1 = axes_bin[0].bar(ringkas_bin['kelompok'], ringkas_bin['jumlah_pelanggan'], color=w_k, alpha=0.88)
for b, v in zip(b1, ringkas_bin['jumlah_pelanggan']):
    axes_bin[0].text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                     f'{v:,}', ha='center', fontsize=9, fontweight='bold')
axes_bin[0].set_ylabel('Jumlah Pelanggan')
axes_bin[0].set_title('Jumlah Pelanggan\nper Kelompok', fontsize=11, fontweight='bold')
axes_bin[0].tick_params(axis='x', rotation=15)
rapikan_ax(axes_bin[0])

# Panel 2
b2 = axes_bin[1].bar(ringkas_bin['kelompok'], ringkas_bin['avg_belanja'], color=w_k, alpha=0.88)
for b, v in zip(b2, ringkas_bin['avg_belanja']):
    axes_bin[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                     f'R${v:,.0f}', ha='center', fontsize=8, fontweight='bold')
axes_bin[1].set_ylabel('Rata-rata Belanja (R$)')
axes_bin[1].set_title('Rata-rata Nilai Belanja\nper Kelompok', fontsize=11, fontweight='bold')
axes_bin[1].tick_params(axis='x', rotation=15)
rapikan_ax(axes_bin[1])

# Panel 3
b3 = axes_bin[2].bar(ringkas_bin['kelompok'], ringkas_bin['avg_ulasan'], color=w_k, alpha=0.88)
for b, v in zip(b3, ringkas_bin['avg_ulasan']):
    axes_bin[2].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                     f'{v:.2f} ⭐', ha='center', fontsize=9, fontweight='bold')
axes_bin[2].set_ylim(0, 5.5)
axes_bin[2].set_ylabel('Rata-rata Skor Ulasan')
axes_bin[2].set_title('Kepuasan per Kelompok\nBelanja', fontsize=11, fontweight='bold')
axes_bin[2].tick_params(axis='x', rotation=15)
rapikan_ax(axes_bin[2])

plt.tight_layout()
st.pyplot(fig_bin)
plt.close()

st.markdown("---")


# ── KESIMPULAN & REKOMENDASI ───────────────────────────────────────────────
st.markdown('<p class="judul-seksi">📝 Kesimpulan & Rekomendasi</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📦 Ketepatan Pengiriman", "💳 Metode Pembayaran", "💡 Rekomendasi"])

with tab1:
    st.markdown("""
    Analisis terhadap seluruh pesanan terkirim Olist menunjukkan mayoritas pesanan berhasil dikirim
    tepat waktu. Namun pesanan yang mengalami keterlambatan memberikan dampak nyata pada kepuasan
    pelanggan — tercermin dari skor ulasan yang lebih rendah secara signifikan dibanding pesanan tepat waktu.

    Rata-rata durasi pengiriman berkisar 12–13 hari dengan variasi tinggi antar wilayah. Negara bagian
    di timur laut dan utara Brasil secara konsisten memiliki durasi pengiriman lebih lama, mengindikasikan
    kebutuhan mendesak untuk memperkuat infrastruktur logistik di wilayah-wilayah tersebut.
    """)

with tab2:
    st.markdown("""
    Kartu kredit mendominasi metode pembayaran dengan pangsa lebih dari 70% transaksi. Ini mencerminkan
    preferensi pelanggan Brasil untuk memanfaatkan fasilitas cicilan yang disediakan oleh kartu kredit.

    Terdapat korelasi positif yang jelas antara jumlah cicilan dan nilai transaksi. Pelanggan yang
    memilih cicilan lebih banyak cenderung bertransaksi dengan nilai yang lebih besar — menunjukkan
    bahwa kemudahan cicilan berfungsi sebagai pendorong pembelian barang bernilai tinggi.
    """)

with tab3:
    st.markdown("""
    - Perkuat jaringan mitra logistik di negara bagian timur laut dan utara untuk menekan angka keterlambatan.
    - Kembangkan program cicilan tanpa bunga dengan tenor lebih panjang untuk mendorong pembelian kategori premium.
    - Fokuskan program loyalitas pada segmen RFM **Berisiko** sebelum beralih menjadi **Hilang**.
    - Rekrut penjual baru di negara bagian dengan efisiensi transaksi tinggi di luar São Paulo.
    - Perbarui sistem estimasi pengiriman berbasis data historis per rute untuk meningkatkan akurasi.
    - Sediakan insentif pembayaran non-kartu kredit (diskon boleto) untuk memperluas basis metode pembayaran.
    """)

st.markdown("---")
st.markdown(
    "<center><small>Dashboard E-Commerce Olist · Abdul Hamid Amin · "
    "Dibangun dengan Streamlit & Matplotlib</small></center>",
    unsafe_allow_html=True
)
