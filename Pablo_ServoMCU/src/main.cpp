#include <Arduino.h>
#include <Wire.h>
#include <U8x8lib.h>
#include <stdarg.h>
#include <math.h>

#include "LSM6DS3.h"

// NRF52 ISR Servo
#define TIMER_INTERRUPT_DEBUG 0
#define ISR_SERVO_DEBUG 0
#include "NRF52_ISR_Servo.h"

// -----------------------------
// Easy mode switch
// -----------------------------
enum class ControlMode : uint8_t { UART, IMU };
static constexpr ControlMode START_MODE = ControlMode::IMU;
volatile ControlMode g_mode = START_MODE;

// -----------------------------
// FSM
// -----------------------------
enum class FsmState : uint8_t { BOOT, RUN, FAILSAFE };
FsmState g_state = FsmState::BOOT;

// -----------------------------
// OLED
// -----------------------------
U8X8_SSD1306_128X64_NONAME_HW_I2C g_oled(U8X8_PIN_NONE, SCL, SDA);

static constexpr uint8_t OLED_COLS = 16;
static constexpr uint8_t OLED_ROWS = 8;

char g_screen[OLED_ROWS][OLED_COLS + 1];
char g_screenPrev[OLED_ROWS][OLED_COLS + 1];

static constexpr uint32_t OLED_UPDATE_MS = 100;
uint32_t g_lastOledUpdateMs = 0;
bool g_heartbeat = false;

static constexpr uint8_t LOG_LINES = 3;
char g_log[LOG_LINES][OLED_COLS + 1];
uint8_t g_logHead = 0;

static void oledClearLine(uint8_t row) {
  g_oled.setCursor(0, row);
  g_oled.print("                ");
}

static void oledWriteLine(uint8_t row, const char* s) {
  oledClearLine(row);
  g_oled.setCursor(0, row);
  g_oled.print(s);
}

static void oledPushLog(const char* msg) {
  strncpy(g_log[g_logHead], msg, OLED_COLS);
  g_log[g_logHead][OLED_COLS] = '\0';
  g_logHead = (uint8_t)((g_logHead + 1) % LOG_LINES);
}

static void oledPushLogFmt(const char* fmt, ...) {
  char buf[64];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);

  char line[OLED_COLS + 1];
  strncpy(line, buf, OLED_COLS);
  line[OLED_COLS] = '\0';
  oledPushLog(line);
}

// -----------------------------
// Pins and UART
// -----------------------------
static constexpr uint8_t SERVO_PIN = D1;
static constexpr uint32_t UART_BAUD = 115200;

// -----------------------------
// Servo safety and behavior
// -----------------------------
static constexpr int SERVO_HARD_MIN_DEG = 0;
static constexpr int SERVO_HARD_MAX_DEG = 180;

// These microseconds define the electrical end stops for the servo pulses.
// For many small servos, something like 800..2450 is reasonable.
// Tune these for your servo if needed.
static constexpr int SERVO_MIN_US = 800;
static constexpr int SERVO_MAX_US = 2450;

int g_softMinDeg = 20;
int g_softMaxDeg = 160;
int g_failsafeDeg = 90;

static constexpr uint32_t SERVO_UPDATE_MS = 10;
static constexpr float SERVO_MAX_DEG_PER_SEC = 120.0f;

int g_servoIndex = -1;
float g_servoCmdDeg = 90.0f;
float g_servoAppliedDeg = 90.0f;
uint32_t g_lastServoUpdateMs = 0;

// -----------------------------
// UART command handling
// -----------------------------
static constexpr uint32_t UART_TIMEOUT_MS = 500;
static constexpr size_t UART_BUF_LEN = 64;

char g_uartBuf[UART_BUF_LEN];
size_t g_uartLen = 0;
uint32_t g_lastUartRxMs = 0;

// -----------------------------
// IMU
// -----------------------------
LSM6DS3 g_imu(I2C_MODE, 0x6A);
bool g_imuOk = false;

static constexpr uint32_t IMU_UPDATE_MS = 20;
uint32_t g_lastImuUpdateMs = 0;

float g_pitchDegFiltered = 0.0f;
static constexpr float PITCH_LPF_ALPHA = 0.15f;
static constexpr float PITCH_INPUT_MIN = -45.0f;
static constexpr float PITCH_INPUT_MAX = 45.0f;

// -----------------------------
// Helpers
// -----------------------------
static float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static int clampServoDeg(int deg) {
  deg = constrain(deg, SERVO_HARD_MIN_DEG, SERVO_HARD_MAX_DEG);
  deg = constrain(deg, g_softMinDeg, g_softMaxDeg);
  return deg;
}

static float mapFloat(float x, float inMin, float inMax, float outMin, float outMax) {
  if (fabsf(inMax - inMin) < 1e-6f) return outMin;
  float t = (x - inMin) / (inMax - inMin);
  t = clampf(t, 0.0f, 1.0f);
  return outMin + t * (outMax - outMin);
}

static bool equalsIgnoreCase(const char* a, const char* b) {
  while (*a && *b) {
    char ca = *a;
    char cb = *b;
    if (ca >= 'a' && ca <= 'z') ca = char(ca - 'a' + 'A');
    if (cb >= 'a' && cb <= 'z') cb = char(cb - 'a' + 'A');
    if (ca != cb) return false;
    ++a;
    ++b;
  }
  return (*a == '\0' && *b == '\0');
}

static void setServoTargetDeg(int deg) {
  g_servoCmdDeg = (float)clampServoDeg(deg);
}

// -----------------------------
// UART protocol
// -----------------------------
static void handleUartLine(const char* line) {
  char tmp[UART_BUF_LEN];
  strncpy(tmp, line, UART_BUF_LEN);
  tmp[UART_BUF_LEN - 1] = '\0';

  char* tok1 = strtok(tmp, " \t\r");
  if (!tok1) return;

  if (equalsIgnoreCase(tok1, "MODE")) {
    char* tok2 = strtok(nullptr, " \t\r");
    if (!tok2) return;

    if (equalsIgnoreCase(tok2, "UART")) {
      g_mode = ControlMode::UART;
      oledPushLog("Mode UART");
    } else if (equalsIgnoreCase(tok2, "IMU")) {
      g_mode = ControlMode::IMU;
      oledPushLog("Mode IMU");
    }
    return;
  }

  if (equalsIgnoreCase(tok1, "LIMIT")) {
    char* tMin = strtok(nullptr, " \t\r");
    char* tMax = strtok(nullptr, " \t\r");
    if (!tMin || !tMax) return;

    int mn = atoi(tMin);
    int mx = atoi(tMax);

    mn = constrain(mn, SERVO_HARD_MIN_DEG, SERVO_HARD_MAX_DEG);
    mx = constrain(mx, SERVO_HARD_MIN_DEG, SERVO_HARD_MAX_DEG);
    if (mn > mx) { int s = mn; mn = mx; mx = s; }

    g_softMinDeg = mn;
    g_softMaxDeg = mx;
    setServoTargetDeg((int)g_servoCmdDeg);

    oledPushLogFmt("Lim %d %d", g_softMinDeg, g_softMaxDeg);
    return;
  }

  if (equalsIgnoreCase(tok1, "FAILSAFE")) {
    char* t = strtok(nullptr, " \t\r");
    if (!t) return;
    g_failsafeDeg = clampServoDeg(atoi(t));
    oledPushLogFmt("FS %d", g_failsafeDeg);
    return;
  }

  int deg = atoi(tok1);
  setServoTargetDeg(deg);
  oledPushLogFmt("Ang %d", clampServoDeg(deg));
}

static void pollUart() {
  while (Serial1.available() > 0) {
    char c = (char)Serial1.read();
    g_lastUartRxMs = millis();

    if (c == '\n') {
      g_uartBuf[g_uartLen] = '\0';
      if (g_uartLen > 0) handleUartLine(g_uartBuf);
      g_uartLen = 0;
    } else if (c != '\r') {
      if (g_uartLen + 1 < UART_BUF_LEN) g_uartBuf[g_uartLen++] = c;
      else g_uartLen = 0;
    }
  }
}

static void updateImuTarget() {
  float ax = g_imu.readFloatAccelX();
  float ay = g_imu.readFloatAccelY();
  float az = g_imu.readFloatAccelZ();

  float pitchRad = atan2f(-ax, sqrtf(ay * ay + az * az));
  float pitchDeg = pitchRad * 57.2957795f;

  g_pitchDegFiltered = (1.0f - PITCH_LPF_ALPHA) * g_pitchDegFiltered + PITCH_LPF_ALPHA * pitchDeg;

  float p = clampf(g_pitchDegFiltered, PITCH_INPUT_MIN, PITCH_INPUT_MAX);
  float target = mapFloat(p, PITCH_INPUT_MIN, PITCH_INPUT_MAX, (float)g_softMinDeg, (float)g_softMaxDeg);

  setServoTargetDeg((int)(target + 0.5f));
}

static void updateServo() {
  if (g_servoIndex < 0) return;

  uint32_t now = millis();
  if (now - g_lastServoUpdateMs < SERVO_UPDATE_MS) return;

  float dt = (now - g_lastServoUpdateMs) / 1000.0f;
  g_lastServoUpdateMs = now;

  float maxStep = SERVO_MAX_DEG_PER_SEC * dt;
  float err = g_servoCmdDeg - g_servoAppliedDeg;

  if (fabsf(err) <= maxStep) g_servoAppliedDeg = g_servoCmdDeg;
  else g_servoAppliedDeg += (err > 0.0f) ? maxStep : -maxStep;

  int outDeg = clampServoDeg((int)(g_servoAppliedDeg + 0.5f));
  NRF52_ISR_Servos.setPosition(g_servoIndex, outDeg);
}

// Build and render 8 text rows
static void updateOledUI() {
  uint32_t now = millis();
  if (now - g_lastOledUpdateMs < OLED_UPDATE_MS) return;
  g_lastOledUpdateMs = now;
  g_heartbeat = !g_heartbeat;

  for (uint8_t r = 0; r < OLED_ROWS; r++) {
    memset(g_screen[r], ' ', OLED_COLS);
    g_screen[r][OLED_COLS] = '\0';
  }

  const char* st = (g_state == FsmState::RUN) ? "RUN" : (g_state == FsmState::FAILSAFE ? "SAFE" : "BOOT");
  snprintf(g_screen[0], OLED_COLS + 1, "ServoCtl %s %c", st, g_heartbeat ? '*' : ' ');

  const char* md = (g_mode == ControlMode::UART) ? "UART" : "IMU";
  snprintf(g_screen[1], OLED_COLS + 1, "Mode %s", md);

  int cmd = clampServoDeg((int)(g_servoCmdDeg + 0.5f));
  int out = clampServoDeg((int)(g_servoAppliedDeg + 0.5f));
  snprintf(g_screen[2], OLED_COLS + 1, "Cmd%3d Out%3d", cmd, out);

  snprintf(g_screen[3], OLED_COLS + 1, "Lim%3d %3d FS%3d", g_softMinDeg, g_softMaxDeg, g_failsafeDeg);

  if (g_mode == ControlMode::UART) {
    uint32_t age = millis() - g_lastUartRxMs;
    if (age > 999) age = 999;
    snprintf(g_screen[4], OLED_COLS + 1, "UART age %3lums", (unsigned long)age);
  } else {
    int p = (int)(g_pitchDegFiltered + (g_pitchDegFiltered >= 0 ? 0.5f : -0.5f));
    snprintf(g_screen[4], OLED_COLS + 1, "Pitch %4d deg", p);
  }

  for (uint8_t i = 0; i < LOG_LINES; i++) {
    uint8_t idx = (uint8_t)((g_logHead + i) % LOG_LINES);
    snprintf(g_screen[5 + i], OLED_COLS + 1, "%-16s", g_log[idx]);
  }

  for (uint8_t r = 0; r < OLED_ROWS; r++) {
    if (strncmp(g_screen[r], g_screenPrev[r], OLED_COLS) != 0) {
      oledWriteLine(r, g_screen[r]);
      strncpy(g_screenPrev[r], g_screen[r], OLED_COLS + 1);
    }
  }
}

static void i2cScanToLog() {
  oledPushLog("I2C scan");
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    uint8_t err = Wire.endTransmission();
    if (err == 0) {
      oledPushLogFmt("I2C 0x%02X", addr);
      delay(50);
    }
  }
}

void setup() {
  Wire.begin();

  g_oled.begin();
  g_oled.setFlipMode(1);
  g_oled.setFont(u8x8_font_chroma48medium8_r);

  i2cScanToLog();

  for (uint8_t r = 0; r < OLED_ROWS; r++) g_screenPrev[r][0] = '\0';
  for (uint8_t i = 0; i < LOG_LINES; i++) snprintf(g_log[i], OLED_COLS + 1, "%-16s", "");

  oledPushLog("Boot");

  Serial1.begin(UART_BAUD);
  g_lastUartRxMs = millis();

  // Servo init via NRF52_ISR_Servo
  g_servoIndex = NRF52_ISR_Servos.setupServo(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
  if (g_servoIndex < 0) {
    oledPushLog("Servo init err");
    g_state = FsmState::FAILSAFE;
  } else {
    oledPushLog("Servo ok");
    setServoTargetDeg(g_failsafeDeg);
    g_servoAppliedDeg = g_servoCmdDeg;
    NRF52_ISR_Servos.setPosition(g_servoIndex, clampServoDeg((int)g_servoAppliedDeg));
  }

  g_imuOk = (g_imu.begin() == 0);
  oledPushLog(g_imuOk ? "IMU ok" : "IMU err");

  g_lastServoUpdateMs = millis();
  g_lastImuUpdateMs = millis();
  g_lastOledUpdateMs = millis();

  if (g_state != FsmState::FAILSAFE) g_state = FsmState::RUN;
  oledPushLog("Ready");
}

void loop() {
  uint32_t now = millis();

  pollUart();

  switch (g_state) {
    case FsmState::BOOT:
      g_state = FsmState::RUN;
      break;

    case FsmState::RUN:
      if (g_mode == ControlMode::UART) {
        if ((now - g_lastUartRxMs) > UART_TIMEOUT_MS) {
          g_state = FsmState::FAILSAFE;
          oledPushLog("UART timeout");
        }
      } else {
        if (!g_imuOk) {
          g_state = FsmState::FAILSAFE;
          oledPushLog("IMU fail");
        } else if (now - g_lastImuUpdateMs >= IMU_UPDATE_MS) {
          g_lastImuUpdateMs = now;
          updateImuTarget();
        }
      }
      updateServo();
      break;

    case FsmState::FAILSAFE:
      setServoTargetDeg(g_failsafeDeg);
      updateServo();

      if (g_mode == ControlMode::UART) {
        if ((now - g_lastUartRxMs) <= UART_TIMEOUT_MS) {
          g_state = FsmState::RUN;
          oledPushLog("UART ok");
        }
      } else {
        if (g_imuOk) {
          g_state = FsmState::RUN;
          oledPushLog("IMU ok");
        }
      }
      break;
  }

  updateOledUI();
}