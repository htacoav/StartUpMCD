# ==========================================
# POR QUE SALIO ESE RESULTADO
#
# Despues de dar la probabilidad, la aplicacion explica que respuestas la
# subieron y cuales la bajaron. Para eso usa valores SHAP.
#
# Que es un valor SHAP
# --------------------
# Se parte de una startup promedio del entrenamiento, que tiene cierta
# probabilidad de fracasar. Despues se mira cuanto se aleja la startup del
# usuario en cada variable, y cuanto mueve eso el resultado. La suma de todos
# esos empujones mas el punto de partida da exactamente la prediccion. Por eso
# es una explicacion honesta y no una aproximacion.
#
# Por que no se usa la libreria shap
# ----------------------------------
# El modelo ganador es una regresion logistica, y para un modelo lineal el valor
# SHAP tiene una formula cerrada:
#
#     SHAP de una variable = peso * (valor de la startup - valor promedio)
#
# Es exactamente lo que calcula shap.LinearExplainer, pero sin sumarle a la
# imagen de Docker una libreria pesada con sus dependencias. Una prueba
# automatica compara estos numeros contra la libreria oficial.
#
# Los numeros estan en log-odds, que es la escala en la que trabaja la
# regresion logistica. No son puntos porcentuales: sirven para comparar que
# variable pesa mas y hacia donde empuja.
# ==========================================

from ml import config

# Nombre legible de cada variable, para mostrarlo en la respuesta de la API
NOMBRES_EN_INGLES = {
    "macro_climate": "Fundraising climate",
    "market_size_score": "Market size",
    "competition_intensity": "Competition",
    "founder_prior_exits": "Prior founding experience",
    "domain_experience_years": "Years in the industry",
    "cofounder_conflict": "Disagreements between founders",
    "team_completeness": "Team coverage",
    "product_market_fit_score": "Paying customers",
    "did_customer_validation": "Customer validation",
    "premature_scaling": "Scaling too early",
    "unit_economics_score": "Margin per sale",
    "marketing_effectiveness": "How customers find you",
    "funding_path": "Funding path",
    "total_raised_usd": "Money raised",
    "monthly_burn_rate": "Monthly spending",
    "runway_months": "Months of runway",
    "industry": "Industry",
    "departamento": "Region",
}


def variable_original(nombre_columna):
    """Devuelve a que variable pertenece una columna del pipeline.

    El pipeline le cambia el nombre a las columnas. Las numericas quedan como
    "numericas__runway_months" y las categoricas se parten en una columna por
    valor, como "categoricas__industry_fintech". Aca se deshace eso para poder
    juntar todas las columnas de industry en una sola explicacion.
    """
    if nombre_columna.startswith("numericas__"):
        return nombre_columna.replace("numericas__", "")

    resto = nombre_columna.replace("categoricas__", "")
    for categorica in config.CATEGORICAS:
        if resto.startswith(categorica + "_"):
            return categorica

    return resto


def explicar(artefacto, fila):
    """Calcula el valor SHAP de cada una de las 17 variables para una startup.

    "fila" es la tabla de una sola fila que se le pasa al modelo.

    Devuelve None si el modelo no es lineal, porque la formula de arriba solo
    vale para modelos lineales. Asi, si un reentrenamiento eligiera un bosque
    aleatorio, la API sigue funcionando y solo deja de dar la explicacion.
    """
    preparacion = artefacto["pipeline"].named_steps["preparacion"]
    modelo = artefacto["pipeline"].named_steps["modelo"]

    if not hasattr(modelo, "coef_") or "shap_background" not in artefacto:
        return None

    columnas = preparacion.get_feature_names_out()
    valores = preparacion.transform(fila)[0]
    promedios = artefacto["shap_background"]
    pesos = modelo.coef_[0]

    # ---------- 1. El punto de partida ----------
    # Es lo que el modelo diria de la startup promedio, en log-odds de fracaso.
    punto_de_partida = float(modelo.intercept_[0])
    for i in range(len(pesos)):
        punto_de_partida += pesos[i] * promedios[i]

    # ---------- 2. El empujon de cada columna ----------
    # Las columnas de una misma variable categorica se suman, porque para el
    # usuario "industry" es una sola respuesta aunque el modelo la vea partida.
    empujones = {}
    for i in range(len(columnas)):
        variable = variable_original(columnas[i])
        shap_de_esta_columna = pesos[i] * (valores[i] - promedios[i])
        empujones[variable] = empujones.get(variable, 0.0) + shap_de_esta_columna

    # ---------- 3. Armar la lista ----------
    # El modelo predice fracaso, pero la pagina habla de supervivencia. Por eso
    # se cambia el signo: un numero positivo significa que esa respuesta sube la
    # probabilidad de sobrevivir.
    factores = []
    for variable in config.CARACTERISTICAS:
        # pandas devuelve numeros de numpy, que la API no sabe convertir a JSON.
        # Por eso se pasan a un float normal de Python. El texto se deja igual.
        valor = fila[variable].iloc[0]
        if not isinstance(valor, str):
            valor = float(valor)

        factores.append({
            "feature": variable,
            "label": NOMBRES_EN_INGLES[variable],
            "value": valor,
            "impact": round(-float(empujones[variable]), 4),
        })

    # Primero las que mas pesan, sin importar si ayudan o perjudican
    factores.sort(key=lambda f: abs(f["impact"]), reverse=True)

    return {
        "method": "SHAP values for a linear model (exact)",
        "units": "log-odds",
        "base_failure_log_odds": round(punto_de_partida, 4),
        "factors": factores,
    }
