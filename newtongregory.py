import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Interpolasi Newton-Gregory", layout="centered")

st.title("🔢 Interpolasi Newton-Gregory")
st.markdown("Masukin data titik-titik yang diketahui, pilih metode interpolasi, terus cari nilai f(x) yang lo butuhin!")

if "reset" not in st.session_state:
    st.session_state.reset = False

# Tombol Reset
if st.button("🔄 Reset Semua Input"):
    st.session_state.reset = True
    st.experimental_rerun()

if not st.session_state.reset:
    st.subheader("📥 Input Data")

    n = st.number_input("Jumlah data (minimal 2)", min_value=2, max_value=20, value=4, step=1, key="n_input")

    with st.form("data_form"):
        x_vals = []
        y_vals = []
        st.markdown("### Isi nilai x dan f(x):")
        for i in range(n):
            col1, col2 = st.columns(2)
            with col1:
                x = st.number_input(f"x{i}", key=f"x{i}")
            with col2:
                y = st.number_input(f"f(x{i})", key=f"y{i}")
            x_vals.append(x)
            y_vals.append(y)
        
        submitted = st.form_submit_button("🔍 Proses Interpolasi")

    if submitted:
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

        if not np.all(np.diff(x_vals) == x_vals[1] - x_vals[0]):
            st.error("❗ Jarak antar x harus sama (uniform).")
        else:
            h = x_vals[1] - x_vals[0]
            df_table = [y_vals.copy()]

            method = st.selectbox("Pilih metode:", ["Newton-Gregory Maju", "Newton-Gregory Mundur"])
            x_interp = st.number_input("Masukkan nilai x yang ingin dicari:", value=float(x_vals[0]), key="x_interp")

            for level in range(1, n):
                prev = df_table[-1]
                diff = [prev[i+1] - prev[i] for i in range(len(prev)-1)]
                df_table.append(np.array(diff))

            # Tampilkan tabel beda hingga
            st.subheader("📊 Tabel Beda Hingga")
            df = pd.DataFrame({f"x{i}": [x_vals[i]] + [row[i] if i < len(row) else None for row in df_table[1:]] for i in range(len(x_vals))})
            st.dataframe(df.transpose(), use_container_width=True)

            # Interpolasi
            if method == "Newton-Gregory Maju":
                p = (x_interp - x_vals[0]) / h
                result = y_vals[0]
                p_term = 1
                for i in range(1, n):
                    p_term *= (p - i + 1)
                    result += (p_term * df_table[i][0]) / math.factorial(i)

            else:
                p = (x_interp - x_vals[-1]) / h
                result = y_vals[-1]
                p_term = 1
                for i in range(1, n):
                    p_term *= (p + i - 1)
                    result += (p_term * df_table[i][-1]) / math.factorial(i)

            st.subheader("🧮 Hasil Interpolasi")
            st.success(f"f({x_interp}) ≈ {result}")

            # Plotting grafik
            st.subheader("📈 Grafik Interpolasi")

            x_plot = np.linspace(min(x_vals), max(x_vals), 200)
            y_plot = []

            for xp in x_plot:
                if method == "Newton-Gregory Maju":
                    p = (xp - x_vals[0]) / h
                    yp = y_vals[0]
                    p_term = 1
                    for i in range(1, n):
                        p_term *= (p - i + 1)
                        yp += (p_term * df_table[i][0]) / math.factorial(i)
                else:
                    p = (xp - x_vals[-1]) / h
                    yp = y_vals[-1]
                    p_term = 1
                    for i in range(1, n):
                        p_term *= (p + i - 1)
                        yp += (p_term * df_table[i][-1]) / math.factorial(i)
                y_plot.append(yp)

            fig, ax = plt.subplots()
            ax.plot(x_plot, y_plot, label="Interpolasi", color="blue")
            ax.scatter(x_vals, y_vals, color="red", label="Titik Data")
            ax.axvline(x_interp, color="green", linestyle="--", label=f"x = {x_interp}")
            ax.scatter([x_interp], [result], color="purple", label="Hasil Interpolasi")
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title("Grafik Interpolasi Newton-Gregory")
            ax.legend()
            st.pyplot(fig)

            # Tombol download grafik
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button("⬇️ Download Grafik", buf.getvalue(), file_name="grafik_interpolasi.png", mime="image/png")
