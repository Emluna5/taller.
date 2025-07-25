import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import integrate
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sympy import symbols, integrate as sp_integrate, lambdify, sympify
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="🧮 Calculadora de Integrales",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .result-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .method-comparison {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .stAlert > div {
        background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🧮 Calculadora de Integrales Interactiva</h1>
    <p>Calcula integrales definidas e indefinidas con visualización gráfica</p>
</div>
""", unsafe_allow_html=True)

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")

# Función matemática
function_input = st.sidebar.text_input(
    "📝 Función f(x):", 
    value="x**2",
    help="Usa sintaxis de Python: x**2, sin(x), exp(-x), sqrt(x), etc."
)

# Tipo de integral
integral_type = st.sidebar.selectbox(
    "📊 Tipo de integral:",
    ["Integral Definida", "Integral Indefinida"]
)

# Configuración para integral definida
if integral_type == "Integral Definida":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        lower_limit = st.number_input("Límite inferior (a):", value=0.0, step=0.1)
    with col2:
        upper_limit = st.number_input("Límite superior (b):", value=2.0, step=0.1)
    
    # Método numérico
    method = st.sidebar.selectbox(
        "🔢 Método numérico:",
        ["Trapecio", "Simpson", "Cuadratura Gaussiana"]
    )
    
    # Número de subdivisiones
    subdivisions = st.sidebar.slider(
        "🔢 Subdivisiones (n):",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )

# Rango para visualización
st.sidebar.subheader("📈 Configuración del gráfico")
col1, col2 = st.sidebar.columns(2)
with col1:
    x_min = st.number_input("x mín:", value=-1.0, step=0.1)
with col2:
    x_max = st.number_input("x máx:", value=3.0, step=0.1)

# Funciones auxiliares
def parse_function(func_str):
    """Convierte string de función a expresión simbólica"""
    try:
        x = symbols('x')
        # Reemplazos comunes
        func_str = func_str.replace('^', '**')
        func_str = func_str.replace('ln', 'log')
        return sympify(func_str), True
    except:
        return None, False

def evaluate_function(func_expr, x_vals):
    """Evalúa la función en puntos dados"""
    try:
        f_lambdified = lambdify(symbols('x'), func_expr, 'numpy')
        return f_lambdified(x_vals)
    except:
        return None

def trapezoidal_rule(func_expr, a, b, n):
    """Regla del trapecio"""
    try:
        f = lambdify(symbols('x'), func_expr, 'numpy')
        x = np.linspace(a, b, n+1)
        y = f(x)
        h = (b - a) / n
        return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    except:
        return None

def simpson_rule(func_expr, a, b, n):
    """Regla de Simpson"""
    try:
        if n % 2 == 1:
            n += 1
        f = lambdify(symbols('x'), func_expr, 'numpy')
        x = np.linspace(a, b, n+1)
        y = f(x)
        h = (b - a) / n
        return h/3 * (y[0] + 4*np.sum(y[1::2]) + 2*np.sum(y[2:-1:2]) + y[-1])
    except:
        return None

def gaussian_quadrature(func_expr, a, b):
    """Cuadratura Gaussiana usando scipy"""
    try:
        f = lambdify(symbols('x'), func_expr, 'numpy')
        result, _ = integrate.quad(f, a, b)
        return result
    except:
        return None

# Procesamiento principal
if st.sidebar.button("🚀 Calcular Integral", type="primary"):
    # Parse de la función
    func_expr, is_valid = parse_function(function_input)
    
    if not is_valid:
        st.error("❌ Error: No se pudo interpretar la función. Verifica la sintaxis.")
    else:
        # Crear layout en columnas
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Resultados")
            
            if integral_type == "Integral Definida":
                if lower_limit >= upper_limit:
                    st.error("❌ El límite inferior debe ser menor que el superior")
                else:
                    # Calcular con diferentes métodos
                    results = {}
                    
                    # Método seleccionado
                    if method == "Trapecio":
                        results['principal'] = trapezoidal_rule(func_expr, lower_limit, upper_limit, subdivisions)
                    elif method == "Simpson":
                        results['principal'] = simpson_rule(func_expr, lower_limit, upper_limit, subdivisions)
                    else:  # Cuadratura Gaussiana
                        results['principal'] = gaussian_quadrature(func_expr, lower_limit, upper_limit)
                    
                    # Calcular también con otros métodos para comparación
                    results['trapecio'] = trapezoidal_rule(func_expr, lower_limit, upper_limit, subdivisions)
                    results['simpson'] = simpson_rule(func_expr, lower_limit, upper_limit, subdivisions)
                    results['gaussiana'] = gaussian_quadrature(func_expr, lower_limit, upper_limit)
                    
                    # Integral simbólica (si es posible)
                    try:
                        x = symbols('x')
                        symbolic_result = sp_integrate(func_expr, (x, lower_limit, upper_limit))
                        results['simbolica'] = float(symbolic_result.evalf())
                    except:
                        results['simbolica'] = None
                    
                    # Mostrar resultado principal
                    if results['principal'] is not None:
                        st.markdown(f"""
                        <div class="result-box">
                            <h3>🎯 {method} (Principal)</h3>
                            <h2 style="color: #667eea;">{results['principal']:.8f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Comparación de métodos
                    st.subheader("🔍 Comparación de Métodos")
                    
                    methods_data = []
                    if results['trapecio'] is not None:
                        methods_data.append({"Método": "Trapecio", "Resultado": f"{results['trapecio']:.8f}"})
                    if results['simpson'] is not None:
                        methods_data.append({"Método": "Simpson", "Resultado": f"{results['simpson']:.8f}"})
                    if results['gaussiana'] is not None:
                        methods_data.append({"Método": "Cuadratura Gaussiana", "Resultado": f"{results['gaussiana']:.8f}"})
                    if results['simbolica'] is not None:
                        methods_data.append({"Método": "Simbólica (Exacta)", "Resultado": f"{results['simbolica']:.8f}"})
                    
                    if methods_data:
                        df_methods = pd.DataFrame(methods_data)
                        st.dataframe(df_methods, use_container_width=True)
                    
                    # Información adicional
                    st.info(f"""
                    **📋 Detalles del cálculo:**
                    - Función: f(x) = {function_input}
                    - Límites: [{lower_limit}, {upper_limit}]
                    - Subdivisiones: {subdivisions}
                    - Método principal: {method}
                    """)
            
            else:  # Integral Indefinida
                try:
                    x = symbols('x')
                    indefinite_result = sp_integrate(func_expr, x)
                    
                    st.markdown(f"""
                    <div class="result-box">
                        <h3>∫ Integral Indefinida</h3>
                        <h3 style="color: #667eea;">∫({function_input})dx = {indefinite_result} + C</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("📝 **Nota:** La constante de integración C se omite en el resultado simbólico.")
                    
                except Exception as e:
                    st.error(f"❌ No se pudo calcular la integral simbólica: {str(e)}")
        
        with col2:
            st.subheader("📈 Visualización")
            
            # Generar puntos para la gráfica
            x_vals = np.linspace(x_min, x_max, 1000)
            
            try:
                y_vals = evaluate_function(func_expr, x_vals)
                
                if y_vals is not None:
                    # Crear gráfica con Plotly
                    fig = go.Figure()
                    
                    # Línea principal de la función
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines',
                        name=f'f(x) = {function_input}',
                        line=dict(color='#667eea', width=3)
                    ))
                    
                    # Si es integral definida, mostrar el área
                    if integral_type == "Integral Definida" and lower_limit < upper_limit:
                        # Área bajo la curva
                        x_area = np.linspace(lower_limit, upper_limit, 200)
                        y_area = evaluate_function(func_expr, x_area)
                        
                        if y_area is not None:
                            fig.add_trace(go.Scatter(
                                x=x_area,
                                y=y_area,
                                fill='tozeroy',
                                mode='none',
                                name='Área bajo la curva',
                                fillcolor='rgba(102, 126, 234, 0.3)'
                            ))
                        
                        # Líneas verticales en los límites
                        y_range = [np.min(y_vals), np.max(y_vals)]
                        
                        fig.add_trace(go.Scatter(
                            x=[lower_limit, lower_limit],
                            y=y_range,
                            mode='lines',
                            name=f'x = {lower_limit}',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[upper_limit, upper_limit],
                            y=y_range,
                            mode='lines',
                            name=f'x = {upper_limit}',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                    
                    # Configurar layout
                    fig.update_layout(
                        title=f'Gráfica de f(x) = {function_input}',
                        xaxis_title='x',
                        yaxis_title='f(x)',
                        template='plotly_white',
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("❌ Error al evaluar la función para la gráfica")
                    
            except Exception as e:
                st.error(f"❌ Error al generar la gráfica: {str(e)}")

# Sección de ayuda
with st.expander("❓ Ayuda y Ejemplos"):
    st.markdown("""
    ### 📝 Sintaxis de funciones:
    - **Operaciones básicas:** `+`, `-`, `*`, `/`, `**` (potencia)
    - **Funciones trigonométricas:** `sin(x)`, `cos(x)`, `tan(x)`
    - **Funciones exponenciales:** `exp(x)`, `log(x)` (logaritmo natural)
    - **Otras funciones:** `sqrt(x)`, `abs(x)`
    - **Constantes:** `pi`, `E` (número e)
    
    ### 🧮 Ejemplos de funciones:
    - `x**2` → Parábola
    - `sin(x)` → Función seno
    - `exp(-x**2)` → Campana de Gauss
    - `x**3 - 2*x + 1` → Polinomio cúbico
    - `1/(1 + x**2)` → Función racional
    - `sqrt(1 - x**2)` → Semicírculo
    
    ### 🔢 Métodos numéricos:
    - **Trapecio:** Aproxima el área usando trapecios
    - **Simpson:** Usa parábolas para mayor precisión
    - **Cuadratura Gaussiana:** Método de alta precisión
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🧮 Calculadora de Integrales | Desarrollado con Streamlit
</div>
""", unsafe_allow_html=True)
