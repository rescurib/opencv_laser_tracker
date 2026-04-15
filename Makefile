# Nombre del proyecto
PROJECT_NAME = kf_laser_tracker

# Archivos fuente
SRC = $(PROJECT_NAME).cpp

# Ejecutable de salida
TARGET = $(PROJECT_NAME)

# Detectar OpenCV usando pkg-config (Linux/Mac) o variables de entorno (Windows)
# Para Windows, se recomienda definir OPENCV_DIR y agregar las rutas de include/lib manualmente si no se usa pkg-config

# Variables para OpenCV (ajustar según tu instalación si es necesario)
# 2>/dev/null redirige stderr a /dev/null si pkg-config no está instalado
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS   = $(shell pkg-config --libs opencv4 2>/dev/null)

# Compilador
CXX = g++
CXXFLAGS = -std=c++11 $(OPENCV_CFLAGS)
LDFLAGS = $(OPENCV_LIBS)

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
