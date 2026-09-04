"""
perfil_arrecife_xbeach.py
-------------------------
Genera un perfil batimétrico 1D sintético tipo "arrecife de fricción"
(talud frontal + laguna/plataforma + playa) y lo exporta en el formato
de grilla y profundidad que requiere XBeach (x.grd, y.grd, *.dep).

Convención de signos
---------------------
z se define como ELEVACIÓN respecto al nivel medio del mar (NMM):
    z > 0  -> por encima del NMM (playa emergida)
    z < 0  -> por debajo del NMM (profundidad)
Esta es la convención posdwn = -1 de XBeach (bathymetry positiva hacia
arriba). Si se usa con posdwn = 1 hay que multiplicar z por -1 antes de
escribir el .dep, o simplemente declarar posdwn = -1 en params.txt
(se imprime automáticamente el bloque sugerido al final).

Autor: generado para flujo de trabajo de Juan Diego (UNAL - Recursos Hídricos)
"""

import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    _TIENE_MPL = True
except ImportError:
    _TIENE_MPL = False


def generar_perfil_arrecife(
    pendiente_playa,
    profundidad_laguna_inicio,
    pendiente_frente_arrecife,
    longitud_laguna,
    profundidad_laguna_fin=None,
    profundidad_offshore=50.0,
    elevacion_final=-2.0,
    ancho_plano_offshore=50.0,
    dx=1.0,
):
    """
    Construye un perfil batimétrico 1D sintético de un arrecife de fricción,
    compuesto por 4 tramos rectos (de mar a tierra):

        1) plano offshore   -> tramo horizontal en z = -profundidad_offshore
                                (requerido por XBeach para la condición de
                                oleaje en el borde mar adentro)
        2) frente de arrecife -> talud que sube desde -profundidad_offshore
                                  hasta -profundidad_laguna_inicio
        3) laguna/plataforma  -> tramo de longitud_laguna que va desde
                                  -profundidad_laguna_inicio hasta
                                  -profundidad_laguna_fin (si ambas
                                  profundidades son iguales, la laguna queda
                                  plana, como antes; si son distintas, la
                                  laguna queda con una pendiente propia)
        4) playa              -> talud que sube desde -profundidad_laguna_fin
                                  hasta elevacion_final

    Parámetros
    ----------
    pendiente_playa : float
        Pendiente de la playa, adimensional (m/m). Ej: 0.05 = pendiente 1:20.
    profundidad_laguna_inicio : float
        Profundidad al INICIO de la laguna (lado mar, donde termina el
        frente del arrecife), respecto al NMM, en metros, como valor
        POSITIVO (ej. 3.0 -> z = -3.0 m).
    pendiente_frente_arrecife : float
        Pendiente del talud frontal del arrecife, adimensional (m/m).
    longitud_laguna : float
        Longitud del tramo de la laguna, en metros.
    profundidad_laguna_fin : float, opcional
        Profundidad al FINAL de la laguna (lado tierra, donde empieza la
        playa), respecto al NMM, en metros, como valor POSITIVO.
        Si se deja en None (default), se usa el mismo valor que
        profundidad_laguna_inicio y la laguna queda plana (comportamiento
        equivalente al de la versión anterior de la función). Si se da un
        valor distinto a profundidad_laguna_inicio, la laguna queda con una
        pendiente propia (pendiente_laguna = |fin - inicio| / longitud_laguna,
        que se calcula automáticamente y se reporta en info_segmentos).
    profundidad_offshore : float, opcional (default 50.0)
        Profundidad del límite oceánico del perfil, valor positivo en metros
        (z = -profundidad_offshore).
    elevacion_final : float, opcional (default -2.0)
        Cota (con signo) del extremo más somero/terrestre del perfil.
        Por defecto -2.0 m (2 m bajo el NMM), tal como se solicitó. Si se
        quiere una playa emergida, usar un valor positivo (ej. +2.0).
    ancho_plano_offshore : float, opcional (default 50.0)
        Longitud del tramo plano inicial en el límite offshore. Puede
        ponerse en 0 si no se necesita, pero XBeach recomienda al menos
        una celda horizontal en el borde de mar adentro.
    dx : float, opcional (default 1.0)
        Resolución espacial (uniforme) del perfil, en metros.

    Retorna
    -------
    x : np.ndarray
        Distancia cross-shore (m), x=0 en el límite offshore.
    z : np.ndarray
        Elevación del fondo (m) respecto al NMM, en cada punto de x.
    info_segmentos : dict
        Diccionario {nombre_tramo: (x_inicio, x_fin)} útil, por ejemplo,
        para luego asignar coeficientes de fricción distintos sobre el
        arrecife/laguna en un archivo de fricción de XBeach. Incluye también:
          - "pendiente_laguna_calculada": pendiente resultante del tramo de
            laguna (m/m).
          - "grid_xbeach": dict con {"nx", "ny", "vardx", "posdwn"}, listos
            para usarse en el bloque de grilla de params.txt.
    """
    if profundidad_laguna_fin is None:
        profundidad_laguna_fin = profundidad_laguna_inicio

    if pendiente_playa <= 0 or pendiente_frente_arrecife <= 0:
        raise ValueError("Las pendientes deben ser valores positivos (m/m).")
    if longitud_laguna < 0 or ancho_plano_offshore < 0:
        raise ValueError("Las longitudes no pueden ser negativas.")
    if longitud_laguna == 0 and profundidad_laguna_inicio != profundidad_laguna_fin:
        raise ValueError(
            "Si profundidad_laguna_inicio y profundidad_laguna_fin son distintas, "
            "longitud_laguna debe ser mayor que 0 (no se puede tener un cambio de "
            "profundidad en una longitud nula)."
        )
    if profundidad_laguna_inicio <= 0 or profundidad_laguna_fin <= 0 or profundidad_offshore <= 0:
        raise ValueError(
            "profundidad_laguna_inicio, profundidad_laguna_fin y profundidad_offshore "
            "deben ser valores positivos (p.ej. 3.0, 1.5 y 50.0 m)."
        )

    z_offshore = -abs(profundidad_offshore)
    z_laguna_inicio = -abs(profundidad_laguna_inicio)
    z_laguna_fin = -abs(profundidad_laguna_fin)

    if z_laguna_inicio <= z_offshore:
        raise ValueError("La laguna debe ser más somera que el límite offshore.")
    if elevacion_final <= z_laguna_fin:
        raise ValueError(
            "elevacion_final debe ser mayor (más somero) que -profundidad_laguna_fin "
            f"(= {z_laguna_fin} m) para que el tramo de playa tenga sentido."
        )

    def tramo(x0, z0, z1, longitud=None, pendiente=None):
        """Genera un tramo recto entre dos cotas (por longitud o por pendiente)."""
        if longitud is None:
            longitud = abs(z1 - z0) / pendiente
        n = max(int(round(longitud / dx)), 1)
        x_seg = x0 + np.linspace(0, longitud, n + 1)
        z_seg = np.linspace(z0, z1, n + 1)
        return x_seg, z_seg

    segmentos = []
    x_actual = 0.0

    # 1) Plano offshore
    if ancho_plano_offshore > 0:
        x_seg, z_seg = tramo(x_actual, z_offshore, z_offshore, longitud=ancho_plano_offshore)
        segmentos.append(("plano_offshore", x_seg, z_seg))
        x_actual = x_seg[-1]

    # 2) Frente del arrecife
    x_seg, z_seg = tramo(x_actual, z_offshore, z_laguna_inicio, pendiente=pendiente_frente_arrecife)
    segmentos.append(("frente_arrecife", x_seg, z_seg))
    x_actual = x_seg[-1]

    # 3) Laguna / plataforma (plana si z_laguna_inicio == z_laguna_fin,
    #    o con su propia pendiente si son distintas)
    pendiente_laguna_calculada = 0.0
    if longitud_laguna > 0:
        x_seg, z_seg = tramo(x_actual, z_laguna_inicio, z_laguna_fin, longitud=longitud_laguna)
        segmentos.append(("laguna", x_seg, z_seg))
        x_actual = x_seg[-1]
        pendiente_laguna_calculada = abs(z_laguna_fin - z_laguna_inicio) / longitud_laguna

    # 4) Playa
    x_seg, z_seg = tramo(x_actual, z_laguna_fin, elevacion_final, pendiente=pendiente_playa)
    segmentos.append(("playa", x_seg, z_seg))

    # Concatenar tramos sin duplicar los puntos de unión
    x_parts = [segmentos[0][1]]
    z_parts = [segmentos[0][2]]
    for _, x_seg, z_seg in segmentos[1:]:
        x_parts.append(x_seg[1:])
        z_parts.append(z_seg[1:])

    x = np.concatenate(x_parts)
    z = np.concatenate(z_parts)

    info_segmentos = {nombre: (float(x_seg[0]), float(x_seg[-1])) for nombre, x_seg, _ in segmentos}
    info_segmentos["pendiente_laguna_calculada"] = pendiente_laguna_calculada
    info_segmentos["grid_xbeach"] = {
        "nx": len(x) - 1,    # número de celdas en x (XBeach: nx = nº de puntos - 1)
        "ny": 0,             # malla 1D
        "vardx": 1,          # grilla no equidistante (se lee de x.grd)
        "posdwn": -1,        # z positivo hacia arriba (NMM = 0), como se generó aquí
    }

    return x, z, info_segmentos


def escribir_archivos_xbeach(x, z, carpeta_salida=".", nombre_dep="bed.dep"):
    """
    Escribe los archivos x.grd, y.grd y el archivo de profundidad (.dep)
    en el formato ASCII que espera XBeach para una malla 1D (ny = 0).

    Parámetros
    ----------
    x, z : np.ndarray
        Vectores devueltos por generar_perfil_arrecife.
    carpeta_salida : str
        Carpeta donde se escriben los archivos (se crea si no existe).
    nombre_dep : str
        Nombre del archivo de profundidad (por convención XBeach suele
        llamarse 'bed.dep').

    Retorna
    -------
    rutas : dict
        Rutas absolutas de los archivos escritos.
    """
    os.makedirs(carpeta_salida, exist_ok=True)
    y = np.zeros_like(x)

    ruta_x = os.path.join(carpeta_salida, "x.grd")
    ruta_y = os.path.join(carpeta_salida, "y.grd")
    ruta_dep = os.path.join(carpeta_salida, nombre_dep)

    # XBeach espera, para una malla 1D (ny=0), una sola fila con (nx+1) valores
    np.savetxt(ruta_x, x.reshape(1, -1), fmt="%.4f")
    np.savetxt(ruta_y, y.reshape(1, -1), fmt="%.4f")
    np.savetxt(ruta_dep, z.reshape(1, -1), fmt="%.4f")

    nx = len(x) - 1
    print(f"Archivos escritos en: {os.path.abspath(carpeta_salida)}")
    print(" -", ruta_x)
    print(" -", ruta_y)
    print(" -", ruta_dep)
    print("\nBloque de grilla sugerido para params.txt:\n")
    print(f"  nx       = {nx}")
    print(f"  ny       = 0")
    print(f"  vardx    = 1")
    print(f"  xfile    = x.grd")
    print(f"  yfile    = y.grd")
    print(f"  depfile  = {nombre_dep}")
    print(f"  posdwn   = -1   # z positivo hacia arriba (NMM = 0), como se generó aquí")

    return {"x.grd": os.path.abspath(ruta_x), "y.grd": os.path.abspath(ruta_y), nombre_dep: os.path.abspath(ruta_dep)}


def graficar_perfil(x, z, info_segmentos, ruta_png=None):
    """Grafica el perfil con los tramos identificados (requiere matplotlib)."""
    if not _TIENE_MPL:
        print("matplotlib no está disponible; se omite el gráfico.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, z, color="black", lw=1.5)
    ax.axhline(0, color="steelblue", lw=0.8, ls="--", label="NMM")

    colores = {
        "plano_offshore": "#cfe8f3",
        "frente_arrecife": "#f6c177",
        "laguna": "#a6d189",
        "playa": "#e5989b",
    }
    for nombre, valor in info_segmentos.items():
        if not (isinstance(valor, tuple) and len(valor) == 2):
            continue  # ej. "pendiente_laguna_calculada", que no es un tramo (x0, x1)
        x0, x1 = valor
        ax.axvspan(x0, x1, color=colores.get(nombre, "lightgray"), alpha=0.4, label=nombre)

    ax.set_xlabel("Distancia cross-shore [m]")
    ax.set_ylabel("Elevación z [m, NMM]")
    ax.set_title("Perfil batimétrico sintético - arrecife de fricción")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if ruta_png:
        fig.savefig(ruta_png, dpi=150)
        print(f"Gráfico guardado en: {os.path.abspath(ruta_png)}")
    return fig


# if __name__ == "__main__":
#     # ------------------------------------------------------------------
#     # Ejemplo de uso con parámetros típicos de un arrecife franjeante
#     # caribeño (ajustar a los valores reales del sitio de estudio).
#     # Aquí la laguna pasa de -3.0 m a -1.5 m en 300 m, es decir, con una
#     # pendiente propia (no plana).
#     # ------------------------------------------------------------------
#     x, z, info = generar_perfil_arrecife(
#         pendiente_playa=0.2,                 # 1:20
#         profundidad_laguna_inicio=5,         # laguna empieza en -3.0 m
#         profundidad_laguna_fin=5,            # laguna termina en -1.5 m
#         pendiente_frente_arrecife=3,        # 1:5
#         longitud_laguna=100,                 # 300 m de laguna
#         profundidad_offshore=20.0,             # borde offshore a -50 m
#         elevacion_final=5,                  # extremo somero en -0.5 m (debe ser
#                                                 # más somero que -profundidad_laguna_fin)
#         ancho_plano_offshore=25.0,
#         dx=1.0,
#     )

#     print("Resumen de tramos (x_inicio, x_fin) [m]:")
#     for nombre, valor in info.items():
#         if isinstance(valor, tuple) and len(valor) == 2:
#             x0, x1 = valor
#             print(f"  {nombre:18s}: {x0:8.1f} -> {x1:8.1f}  (long. {x1 - x0:7.1f} m)")
#     print(f"\nPendiente de la laguna calculada: {info['pendiente_laguna_calculada']:.4f} m/m")
#     print(f"Parámetros de grilla para params.txt: {info['grid_xbeach']}")
#     print(f"Número total de puntos: {len(x)}")
#     print(f"x total del perfil: 0 -> {x[-1]:.1f} m")
#     print(f"z mínimo: {z.min():.2f} m | z máximo: {z.max():.2f} m")

#     escribir_archivos_xbeach(x, z, carpeta_salida=path_case+"input")
#     graficar_perfil(x, z, info, ruta_png=path_case+"input/perfil_arrecife.png")