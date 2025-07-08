import csv

input_file = "productos_cultivo_piura.csv"
output_file = "productos_limpio.csv"

# Encabezado correcto
header = ['FECHA_CORTE','FECHA_MUESTRA','DEPARTAMENTO','PROVINCIA','DISTRITO','UBIGEO',
          'ANO','MES','COD_CULTIVO','CULTIVO','SIEMBRA','COSECHA','PRODUCCION','VERDE_ACTUAL','PRECIO_CHACRA']

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    writer.writerow(header)  # Escribir encabezado fijo

    for row in reader:
        # Saltar encabezado original si está repetido
        if row[0] == 'FECHA_CORTE':
            continue

        # Caso correcto: 15 columnas
        if len(row) == 15:
            writer.writerow(row)
            continue

        # Caso incorrecto: más de 15 columnas
        elif len(row) > 15:
            fixed_row = row[:9]  # FECHA_CORTE hasta COD_CULTIVO
            cultivo_parts = []

            # Desde la posición 9, buscamos juntar CULTIVO hasta que aparezca un número (probable SIEMBRA)
            i = 9
            while i < len(row):
                value = row[i].strip()
                try:
                    float(value)  # si se puede convertir, es dato numérico
                    break
                except ValueError:
                    cultivo_parts.append(value)
                    i += 1

            cultivo = ",".join(cultivo_parts)
            fixed_row.append(cultivo)

            remaining = row[i:i+5]
            if len(remaining) == 5:
                fixed_row.extend(remaining)
                writer.writerow(fixed_row)

# Confirmación
print(f"Archivo limpio guardado como '{output_file}'")
