# Hardware Wiring Guide — AI Blind Assistant

Complete GPIO wiring reference for Raspberry Pi 5 (also compatible with Pi 4).

---

## GPIO Pin Map (BCM numbering)

```
Raspberry Pi 5 — GPIO Header (40-pin)

         3.3V ● ● 5V          Pin 1 / Pin 2
      GPIO 2 ● ● 5V          Pin 3 / Pin 4
      GPIO 3 ● ● GND         Pin 5 / Pin 6
      GPIO 4 ● ● GPIO 14     Pin 7 / Pin 8  (GPS TX→Pi RX)
         GND ● ● GPIO 15     Pin 9 / Pin 10 (GPS RX←Pi TX)
     GPIO 17 ● ● GPIO 18     Pin 11 / Pin 12  ← SOS Button
     GPIO 27 ● ● GND         Pin 13 / Pin 14
     GPIO 22 ● ● GPIO 23     Pin 15 / Pin 16  ← TRIG
         3.3V ● ● GPIO 24     Pin 17 / Pin 18  ← ECHO
     GPIO 10 ● ● GND         Pin 19 / Pin 20
      GPIO 9 ● ● GPIO 25     Pin 21 / Pin 22
     GPIO 11 ● ● GPIO 8      Pin 23 / Pin 24
         GND ● ● GPIO 7      Pin 25 / Pin 26
      GPIO 0 ● ● GPIO 1      Pin 27 / Pin 28
      GPIO 5 ● ● GND         Pin 29 / Pin 30
      GPIO 6 ● ● GPIO 12     Pin 31 / Pin 32
     GPIO 13 ● ● GND         Pin 33 / Pin 34
     GPIO 19 ● ● GPIO 16     Pin 35 / Pin 36
     GPIO 26 ● ● GPIO 20     Pin 37 / Pin 38
         GND ● ● GPIO 21     Pin 39 / Pin 40
```

---

## HC-SR04 Ultrasonic Sensor

| HC-SR04 Pin | Raspberry Pi Pin | BCM | Note                              |
|-------------|-----------------|-----|-----------------------------------|
| VCC         | Pin 2 (5V)      | —   | Sensor requires 5V                |
| GND         | Pin 6 (GND)     | —   | Common ground                     |
| TRIG        | Pin 16          | 23  | Trigger pulse output from Pi      |
| ECHO        | Pin 18          | 24  | Echo input — **USE VOLTAGE DIVIDER** |

> [!WARNING]
> The ECHO pin outputs 5V but Raspberry Pi GPIO is 3.3V tolerant only.
> Use a voltage divider: 1kΩ resistor from ECHO to GPIO, 2kΩ from GPIO to GND.

### Voltage Divider Circuit:
```
HC-SR04 ECHO ──── 1kΩ ──── GPIO24 (Pi)
                               │
                             2kΩ
                               │
                             GND
```

---

## NEO-6M GPS Module

| NEO-6M Pin | Raspberry Pi Pin | BCM     | Note                        |
|------------|------------------|---------|-----------------------------|
| VCC        | Pin 1 (3.3V)    | —       | Module runs on 3.3V         |
| GND        | Pin 9 (GND)     | —       | Common ground               |
| TX         | Pin 10          | GPIO 15 | GPS data → Pi RX (UART RX)  |
| RX         | Pin 8           | GPIO 14 | Pi TX → GPS (optional)      |

> [!IMPORTANT]
> You MUST disable the serial console before connecting the GPS module.
> Run: `sudo raspi-config` → Interface Options → Serial Port → No (login shell) → Yes (hardware)

---

## Microphone (USB or I2S)

### Option A: USB Microphone (Easiest)
- Plug into any USB port
- Set `AUDIO_INPUT_DEVICE: -1` in config (auto-detect)
- No wiring required

### Option B: I2S MEMS Microphone (e.g., INMP441)

| INMP441 Pin | Raspberry Pi Pin | BCM     |
|-------------|-----------------|---------|
| VDD         | Pin 1 (3.3V)   | —       |
| GND         | Pin 6 (GND)    | —       |
| SD (data)   | Pin 38          | GPIO 20 |
| WS (word select) | Pin 35    | GPIO 19 |
| SCK (clock) | Pin 40          | GPIO 21 |
| L/R         | GND            | —       | Set to mono left channel |

---

## Speaker (3.5mm or USB Audio)

### Option A: 3.5mm Audio Jack (Built-in)
- Connect speaker to Pi 3.5mm jack
- Run: `amixer cset numid=3 1` (force 3.5mm output)

### Option B: USB Audio DAC (Recommended for better quality)
- Plug USB audio adapter into any USB port
- Automatically detected by ALSA

### Option C: I2S Speaker (MAX98357A)

| MAX98357A Pin | Raspberry Pi Pin | BCM     |
|---------------|-----------------|---------|
| VIN           | Pin 2 (5V)     | —       |
| GND           | Pin 6 (GND)    | —       |
| DIN (data)    | Pin 40          | GPIO 21 |
| BCLK (clock)  | Pin 12          | GPIO 18 |
| LRC (L/R)     | Pin 35          | GPIO 19 |

---

## SOS Emergency Button (Optional)

| Button Terminal | Raspberry Pi Pin | BCM     |
|-----------------|-----------------|---------|
| Terminal 1      | Pin 11          | GPIO 17 |
| Terminal 2      | Pin 14 (GND)   | —       |

The button uses the internal pull-up resistor (no external resistor needed).
Hold for 3 seconds to trigger SOS.

---

## Pi Camera Module 3

Connect to the Camera Serial Interface (CSI) connector on the Raspberry Pi.
Use the 15-pin FFC ribbon cable.

> **Note:** If using two cameras, use the second CSI port. Enable camera in:
> `sudo raspi-config` → Interface Options → Camera → Enable

---

## Complete Wiring Diagram

```
                        ┌──────────────────────────┐
 USB Microphone ────────┤ USB Port A               │
 USB Audio DAC  ────────┤ USB Port B               │
 Pi Camera FFC  ────────┤ CSI Connector            │
                        │                          │
 HC-SR04 VCC ───────────┤ Pin 2  (5V)              │
 NEO-6M VCC ────────────┤ Pin 1  (3.3V)            │
 HC-SR04 GND ───────────┤ Pin 6  (GND)             │
 NEO-6M GND ────────────┤ Pin 9  (GND)             │
 NEO-6M TX ─────────────┤ Pin 10 (GPIO15/RXD)      │
 NEO-6M RX ─────────────┤ Pin 8  (GPIO14/TXD)      │
 SOS Button 1 ──────────┤ Pin 11 (GPIO17)          │
 SOS Button 2 ──────────┤ Pin 14 (GND)             │
 HC-SR04 TRIG ──────────┤ Pin 16 (GPIO23)          │
 HC-SR04 ECHO ──1kΩ────┤ Pin 18 (GPIO24)          │
              └──2kΩ──GND                           │
                        └──────────────────────────┘
```
