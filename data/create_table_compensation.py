import pandas as pd

# Crear la tabla de datos unificados
data = {
    "description": [
        "Humedad leve (pequeñas manchas o manchas superficiales)",
        "Humedad moderada (manchas visibles, pero sin daños estructurales)",
        "Humedad severa (daños a paredes, techos o muebles)",
        "Humedad extrema (infiltración de agua significativa, daños estructurales graves)",
        "Humedad por filtraciones en sótano (daños en cimientos o estructura)",
        "Madera afectada por humedad (degradación o pudrición)",
        "Moho o hongos debido a humedad persistente)"
    ],
    "compensation": [
        "500€ - 2,500€",
        "1,000€ - 5,000€",
        "3,000€ - 10,000€",
        "6,000€ - 25,000€",
        "4,000€ - 15,000€",
        "2,000€ - 10,000€",
        "1,000€ - 8,000€"
    ]
}

# Crear el dataframe
#df = pd.DataFrame(data)

# Guardar como archivo CSV
#csv_file_path = "data/compensation_table.csv"
#df.to_csv(csv_file_path, index=False)

import random

# Lista de 25 descripciones base con más variedad
base_descriptions = [
    "Pequeñas manchas de humedad en la pared del dormitorio. La pintura comienza a abombarse y hay un leve cambio de color en la superficie. No hay moho visible, pero el olor a humedad es perceptible.",
    "Filtración de agua en el techo de la cocina. Se observan gotas acumuladas y la pintura empieza a desprenderse. La humedad se extiende lentamente hacia las paredes cercanas.",
    "Manchas de moho negro en las esquinas del baño. Las juntas de los azulejos están ennegrecidas y la ventilación deficiente ha intensificado el problema. Se nota un fuerte olor a humedad.",
    "Techo de yeso con pintura descascarada por filtraciones. La humedad ha debilitado la estructura y se observan grietas en la superficie. Hay riesgo de que parte del techo se desprenda.",
    "Armarios de madera en la cocina afectados por humedad. La superficie está hinchada y comienza a desprender un olor a moho. Las bisagras metálicas presentan signos de oxidación.",
    "Sótano con paredes cubiertas de manchas de humedad. El suelo está húmedo y algunos muebles almacenados han comenzado a deteriorarse. Hay presencia de moho blanco en las esquinas.",
    "Humedad en vigas de madera del techo. La madera se ve oscura y frágil, con signos de pudrición. Si no se trata, puede comprometer la estructura de la vivienda.",
    "Goteras en el garaje de una casa. El agua cae directamente sobre el capó del coche y ha dejado manchas en el suelo. En las paredes, la humedad ha generado moho verde.",
    "Moqueta de oficina con manchas de humedad. La tela ha absorbido la filtración y el olor a moho es fuerte. Algunas sillas metálicas cercanas muestran signos de oxidación.",
    "Pared exterior de una fábrica con manchas verdosas por filtraciones. La pintura se ha desprendido y el material de la fachada comienza a deteriorarse.",
    "Humedad en un almacén comercial. Las cajas de cartón almacenadas han comenzado a debilitarse por la humedad. Se observan hongos en las paredes y un olor desagradable en la zona.",
    "Grietas en el techo de una nave industrial por filtración de agua. La humedad ha debilitado las estructuras metálicas, aumentando el riesgo de corrosión.",
    "Filtraciones en las ventanas de un restaurante. El agua se acumula en los marcos y ha comenzado a gotear sobre las mesas. La madera de los muebles muestra signos de hinchazón.",
    "Tuberías en mal estado provocando humedad en un baño. La pintura de la pared se está desprendiendo y las baldosas están perdiendo adherencia.",
    "Filtración de agua en una bodega subterránea. Las paredes de piedra están húmedas y algunas botellas han sido afectadas por moho.",
    "Vestuario de gimnasio con paredes mohosas. La mala ventilación ha provocado la proliferación de hongos y el mal olor es evidente.",
    "Techo de oficina con manchas de humedad alrededor de las lámparas. El material de yeso comienza a agrietarse y se observan gotas en algunas zonas.",
    "Humidificación excesiva en una librería. Algunos libros tienen páginas arrugadas y cubiertas con manchas de moho.",
    "Escenario de teatro con telones afectados por humedad. La tela tiene manchas oscuras y la madera del piso está hinchada.",
    "Humedad en una escuela. Las paredes de las aulas tienen grandes manchas de moho y la pintura está descascarada.",
    "Cámara frigorífica con condensación excesiva. El agua acumulada en el suelo ha generado resbalones y algunos productos han sido afectados.",
    "Taller mecánico con filtraciones en el techo. Se observan charcos en el suelo y herramientas con signos de corrosión.",
    "Filtración de agua en un hospital. El pasillo subterráneo presenta paredes mojadas y charcos en el suelo. Hay riesgo de afectación al sistema eléctrico.",
    "Edificio de apartamentos con filtraciones en el garaje. Las paredes están cubiertas de manchas verdosas y el suelo presenta charcos constantes.",
    "Supermercado con goteras en la zona de almacenamiento. Se han tenido que desechar cajas de productos afectados por la humedad.",
    "Casa rural con cimientos afectados por la humedad. Se observan grietas en las paredes y las juntas de piedra han comenzado a deshacerse.",
    "Hotel con problemas de humedad en habitaciones. Algunas paredes tienen moho visible y el olor a humedad es fuerte.",
    "Fábrica con maquinaria afectada por humedad ambiental. Algunas piezas metálicas presentan corrosión avanzada y requieren mantenimiento.",
    "Gimnasio con espejos empañados y moho en las esquinas. La mala ventilación ha provocado la acumulación de humedad en el ambiente.",
    "Nave industrial con suelo de cemento resquebrajado por filtraciones. En algunas zonas, la humedad ha debilitado la estructura.",
    "Apartamento con suelo de parquet hinchado por filtraciones. La madera ha comenzado a levantarse y el problema se extiende por varias habitaciones.",
    "Escalera de emergencia de metal oxidada por la humedad. La estructura se ve debilitada y algunas zonas presentan corrosión avanzada.",
    "Techo de almacén con goteras constantes. El agua cae sobre estanterías de productos, poniendo en riesgo la mercancía almacenada.",
    "Bodega de vino con problemas de humedad en las barricas. La madera ha empezado a deteriorarse y algunas botellas muestran signos de afectación.",
    "Centro comercial con filtraciones en el techo. La humedad ha dejado manchas oscuras y algunas zonas presentan desprendimiento de pintura.",
    "Aeropuerto con goteras en la terminal de embarque. El agua se acumula en el suelo y algunas estructuras metálicas muestran corrosión.",
    "Residencia de ancianos con humedad en las habitaciones. El moho ha aparecido en algunas paredes y hay olor persistente a humedad.",
    "Biblioteca con documentos antiguos afectados por la humedad. Las páginas están amarillentas y algunas se han pegado entre sí.",
    "Clínica veterinaria con filtraciones en la zona de quirófano. La humedad amenaza la esterilidad del área y se requieren reparaciones inmediatas.",
    "Universidad con techos dañados por filtraciones. En algunas aulas, la humedad ha provocado el desprendimiento de pintura y moho en las esquinas.",
    "Humedad en una tienda de ropa. Los estantes de madera están hinchados y algunas prendas muestran signos de moho.",
    "Edificio gubernamental con filtraciones en el techo de los despachos. El agua cae sobre escritorios y documentos importantes.",
    "Planta de producción con humedad en el suelo. Se han formado charcos en zonas críticas, afectando la seguridad del personal.",
    "Centro deportivo con humedad en los vestuarios. Las taquillas metálicas están oxidadas y las paredes presentan hongos visibles.",
    "Oficinas con paredes cubiertas de moho. El problema se ha extendido y algunos equipos electrónicos comienzan a verse afectados.",
    "Humedad en una galería de arte. Algunas obras han sido dañadas por la alta humedad del ambiente.",
    "Centro de datos con humedad afectando servidores. El riesgo de fallos eléctricos es elevado y se requieren soluciones inmediatas."
]
# Crear dataframe
df_images_updated = pd.DataFrame({"description": base_descriptions})

# Guardar como CSV
#csv_images_updated_path = "data/descriptions.csv"
#df_images_updated.to_csv(csv_images_updated_path, index=False)

df = pd.read_csv("data/descriptions.csv")

# Guardarlo en formato JSON
df.to_json("data/descriptions.json", orient="records", force_ascii=False, indent=4)
