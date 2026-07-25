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

1. Railway → tu servicio → **Variables** → `MONEYCLUSH_DB` = `/data/moneyclush.db`
2. Railway → **Volumes** → *New Volume*, punto de montaje `/data`

Sin `MONEYCLUSH_DB` la cabecera muestra `DB EFÍMERA` en rojo y se emite una
alerta al arrancar. El fichero `data/predictions.jsonl` se mantiene como
registro en texto plano y se migra automáticamente a SQLite al iniciar.
