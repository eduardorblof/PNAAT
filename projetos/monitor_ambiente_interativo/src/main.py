import time
import dht
from machine import Pin, I2C, ADC
from ssd1306 import SSD1306_I2C # Importa a classe da biblioteca que adicionamos

# --- Definição de limite para o led ---
led_limit = 24

# --- Inicialização do led ---
led_pin = Pin(12, Pin.OUT)

# --- Inicialização do potenciometro ---
pot_pin = Pin(35)
pot_adc = ADC(pot_pin)
pot_adc.atten(ADC.ATTN_11DB)

# --- Inicialização do Sensor de Temperatura ---E
dht = dht.DHT22(Pin(13))

# --- Inicialização do I2C ---
i2c = I2C(0, scl=Pin(22), sda=Pin(21))

# --- Inicialização do Display ---
# Define a largura e altura do display em pixels
largura_oled = 128
altura_oled = 64
# Cria o objeto oled, passando as dimensões e o objeto i2c
oled = SSD1306_I2C(largura_oled, altura_oled, i2c)

print("Inicializando o projeto...")

while True:
    # --- Passo 1: Comandar a medição ---
    # Este comando faz o ESP32 "acordar" o sensor e ler o fluxo de 40 bits.
    dht.measure()
    
    # --- Passo 2: Obter os valores ---
    # Após a medição, podemos acessar os valores processados.
    temperatura = dht.temperature()  # Retorna a temperatura em Celsius
    umidade = dht.humidity()       # Retorna a umidade relativa em %

    # --- Passo 3: Leitura do potenciometro
    pot_valor = pot_adc.read()

    # --- Passo 4: Mapeia o valor para o limite do sensor de temperatura
    led_limit = ((pot_valor - 0) * (80 - -40) / (4096 - 0) + (-40))

    # --- Passo 5: Calcula limite para o led ---
    if temperatura >= led_limit:
        led_pin.value(1)
    else:
        led_pin.value(0)

    # --- Passo 6: Atualiza o display ---
    oled.fill(0) # Apaga a área do contador

    oled.text("Temp: " + str(temperatura) + " C", 0, 0)
    oled.text("Hum: " + str(umidade) + " %", 0, 16)
    oled.text("Limit: " + str(led_limit) + " c", 0, 40)

    oled.show()

    time.sleep(1)

