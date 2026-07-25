# Despliegue en trd.asto.work

El dashboard corre en modo **paper**: lee datos públicos de Polymarket y no
envía órdenes ni guarda claves. Aun así queda detrás de basic auth, porque
muestra tu actividad y porque el día que añadas claves reales el panel ya
estará protegido.

## Antes de empezar

1. Apunta el DNS de `trd.asto.work` (registro A) a la IP del VPS.
2. Abre los puertos 80 y 443 en el firewall.
3. Comprueba que resuelve: `dig +short trd.asto.work`

Si el DNS no resuelve todavía, certbot fallará.

## Instalación

Sube el proyecto al servidor y ejecuta el instalador:

```bash
scp -r D:/Anthony/Projects/MoneyClush root@TU_IP:/opt/moneyclush
```

```bash
ssh root@TU_IP "cd /opt/moneyclush && sudo bash deploy/install.sh"
```

El script instala dependencias, crea el usuario de servicio, levanta systemd,
te pide usuario y contraseña para el panel, configura nginx y saca el
certificado TLS con certbot.

Para usar otro dominio: `sudo DOMAIN=otro.dominio bash deploy/install.sh`

## Operación

```bash
sudo systemctl status moneyclush
```

```bash
sudo journalctl -u moneyclush -f
```

Reiniciar tras actualizar el código:

```bash
sudo systemctl restart moneyclush
```

## Métricas

Cada resolución de ventana se graba en `/opt/moneyclush/data/paper_metrics.jsonl`.
Para descargarlas y analizarlas en local:

```bash
scp root@TU_IP:/opt/moneyclush/data/paper_metrics.jsonl ./data/
```

## Backtest en el servidor

El histórico se descarga y valida con los mismos scripts:

```bash
cd /opt/moneyclush && sudo -u moneyclush .venv/bin/python scripts/fetch_history.py --windows 1000
```

```bash
cd /opt/moneyclush && sudo -u moneyclush .venv/bin/python scripts/backtest_real.py
```

Dejarlo acumulando días de histórico es justo lo que le falta al análisis:
con 300 ventanas el sesgo del favorito sale a p=0.08, sin significancia.

## Nota de seguridad

`TRADING_MODE` está en `paper` por defecto. No pongas `live` ni claves de API
en el `.env` del servidor hasta que el paper trading acumule varios cientos de
resoluciones y confirmes que el modelo gana de verdad. El backtest actual dice
que **no** las tiene todavía.

## Persistencia (obligatorio en Railway)

El sistema acumula predicciones y resultados durante semanas para poder
concluir sobre el sesgo del favorito. El disco de un contenedor se borra en
cada despliegue, así que sin un volumen se pierde todo.

Hacen falta **los dos pasos**, y el orden importa poco pero omitir el segundo
es el error silencioso:

1. Railway → **Volumes** → *New Volume*, punto de montaje `/data`
2. Railway → tu servicio → **Variables** → `MONEYCLUSH_DB` = `/data/moneyclush.db`

Definir la variable sin montar el volumen **no da ningún error**: `/data` se
crea como una carpeta normal del contenedor, todas las escrituras funcionan y
los datos se borran igual en el siguiente despliegue. Por eso el panel no se
fía de la variable, sino que lo comprueba:

| Cabecera | Significado |
|---|---|
| *(sin badge)* | `durable` — verificado, los datos ya sobrevivieron a un redeploy |
| `DB NO VERIF.` | aún no ha pasado un despliegue; no se puede afirmar nada todavía |
| `DB EFÍMERA` | `/data` no es un punto de montaje, o falta la variable — se pierden |

Cada arranque escribe una fila en la tabla `boots` con el id del despliegue.
En cuanto la base contiene arranques de un despliegue **anterior y distinto**,
la persistencia queda demostrada y el badge desaparece. Pasa el ratón por
encima para ver el motivo exacto.

Los ficheros de datos (`data/*.db`, `data/predictions.jsonl`) están fuera de
git a propósito: si se commitean, cada despliegue sobrescribe el histórico
acumulado con la copia del repositorio.
