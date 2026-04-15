#include <opencv2/opencv.hpp>
#include <iostream>

// EWMA (Exponentially Weighted Moving Average) como modelo de fondo simple

int main(int argc, char** argv) {
    int cam_index = 0;
    if (argc > 1) {
        cam_index = std::stoi(argv[1]);
    }

    // Elegir backend de video según plataforma
#ifdef _WIN32
    // Windows (DirectShow, MSMF o cualquiera por defecto)
    cv::VideoCapture cap(cam_index,cv::CAP_DSHOW);
#else
    // Linux (usar V4L2 para evitar problemas con GStreamer)
    cv::VideoCapture cap(cam_index, cv::CAP_V4L2);
#endif

    if (!cap.isOpened()) {
        std::cerr << "No se pudo abrir la cámara con índice " << cam_index << std::endl;
        return 1;
    }

    // --- Parámetros ajustables principales ---
    // Tasa de aprendizaje para el modelo de fondo EWMA (promedio móvil exponencial)
    // Valores bajos = adaptación lenta, valores altos = adaptación rápida a cambios de iluminación
    double alpha = 0.01; // Valor típico: 0.01 a 0.1
    int alpha_counter = 0; // Contador para acelerar temporalmente la adaptación

    // Umbral dinámico para la detección de foreground
    double threshold;
    cv::Scalar scene_mean, scene_stddev;
    cv::Scalar old_scene_mean, old_scene_stddev;
    double min_threshold = 30.0; // Umbral mínimo para evitar ruido excesivo (ajustar según ruido de la cámara)
    cv::Scalar mean, stddev;
    cv::Mat currentFrame, grayFrame, diff;
    cv::Mat background; // Acumulador CV_32F para el fondo
    cv::Mat hsvFrame;
    cv::Mat brightMask;
    cv::Mat valueMask;
    cv::Mat detectionMask;

    std::cout << "Presiona ESC para salir.\n";
    cv::Mat frame;

    // --- Inicialización del Filtro de Kalman ---
    // Estado: [x, y, dx, dy] donde (x, y) es la posición y (dx, dy) la velocidad estimada
    cv::KalmanFilter KF(4, 2, 0);

    // Matriz de transición de estado: modelo de velocidad constante
    KF.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);

    // Matriz de observación: medimos x e y directamente, no la velocidad
    // CORRECCIÓN: definida explícitamente en lugar de setIdentity()
    KF.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    // Covarianza del ruido de proceso
    // Q_vel >> Q_pos para que la velocidad pueda cambiar rápido
    // Aumentamos el ruido de velocidad para que la predicción sea más "futura"
    KF.processNoiseCov = (cv::Mat_<float>(4, 4) <<
        1e-2, 0,    0,    0,
        0,    1e-2, 0,    0,
        0,    0,    20.0, 0,
        0,    0,    0,   20.0);

    // Covarianza del ruido de medición
    // Sin cambios: 1e-1 es razonable para un centroide HSV/R-B
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));

    // Covarianza inicial del error
    // CORRECCIÓN: alta incertidumbre solo en velocidad, no en posición
    setIdentity(KF.errorCovPost, cv::Scalar::all(1));
    KF.errorCovPost.at<float>(2, 2) = 1e2;
    KF.errorCovPost.at<float>(3, 3) = 1e2;

    // Estado inicial: se establece con la primera medición válida (ver bucle principal)
    //cv::randn(KF.statePost, cv::Scalar::all(0), cv::Scalar::all(0.1));

    cv::Mat measurement = cv::Mat::zeros(2, 1, CV_32F); // Medición: posición (x, y)
    bool kf_initialized = false; // Se inicializa con la primera medición válida

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "No se pudo capturar el frame." << std::endl;
            break;
        }

        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // Calcular iluminacion media del frame (útil para adaptar el umbral y detectar cambios bruscos)
        cv::meanStdDev(grayFrame, scene_mean, scene_stddev);

        // Inicializar el fondo con el primer frame
        if (background.empty()) 
        {
            // Primer frame: se usa como fondo inicial
            grayFrame.convertTo(background, CV_32F);
        } else
        {
            // Actualizar el fondo con EWMA
            // Si hay un cambio brusco de iluminación, se acelera temporalmente la adaptación
            double mean_diff = cv::abs(scene_mean[0] - old_scene_mean[0]);
            if(mean_diff > 50) 
            { // Cambio brusco de luz: adapta rápido
                alpha_counter = 30;
                alpha = 0.5; // Learning rate alto temporalmente
            }

            if(alpha_counter > 0) 
            {
                alpha_counter--;
                if(alpha_counter == 0) 
                {
                    alpha = 0.01; // Vuelve al learning rate normal
                }
            }

            cv::accumulateWeighted(grayFrame, background, alpha);
        }

        // Calcular la diferencia absoluta entre el frame actual y el fondo
        // (detección de movimiento/transientes)
        cv::Mat bgU8;
        background.convertTo(bgU8, CV_8U);

        // Suavizado para reducir ruido antes de comparar
        cv::GaussianBlur(grayFrame, grayFrame, cv::Size(7, 7), 1.5);
        cv::GaussianBlur(bgU8, bgU8, cv::Size(7, 7), 1.5);
        cv::absdiff(grayFrame, bgU8, diff);


        // Umbral dinámico: media + k*sigma de la diferencia
        // k controla la sensibilidad a transientes (valores típicos: 1 a 3)
        cv::meanStdDev(diff, mean, stddev);
        threshold = mean[0] + 1 * stddev[0]; // k=1 (ajustar para más/menos sensibilidad)

        // Ajustar umbral mínimo según la iluminación de la escena
        // Para evitar que el umbral baje demasiado en escenas oscuras
        min_threshold = 25 + (1 / scene_mean[0]); 
        threshold = std::max(threshold, min_threshold);

        cv::Mat foregroundMask;
        cv::threshold(diff, foregroundMask, threshold, 255, cv::THRESH_BINARY);

        // Mostrar resultado enmascarado en el frame con color original
        cv::Mat result = cv::Mat::zeros(frame.size(), frame.type());
        frame.copyTo(result, foregroundMask);

        // Convert ROI into HSV to visualize the foreground mask better
        cv::cvtColor(result, hsvFrame, cv::COLOR_BGR2HSV);

        // Threshold the saturation and value channels to detect bright objects
        std::vector<cv::Mat> hsvChannels;
        cv::split(hsvFrame, hsvChannels);
        
        // Value masking: umbral en el canal de valor (brillo)
        // 180 es el umbral de brillo (ajustar según la potencia del láser y la cámara)
        cv::threshold(hsvChannels[2], valueMask, 180, 255, cv::THRESH_BINARY); // Value threshold

        //cv::bitwise_and(detectionMask, brightMask, brightMask);
        result = cv::Mat::zeros(frame.size(), frame.type());
        frame.copyTo(result, valueMask);


        // Compute moments to get center of mass of the detected bright areas
        cv::Moments m = cv::moments(valueMask, true);

        // --- Filtro de Kalman ---
        // Predicción: dónde espera el filtro que esté el láser
        cv::Mat prediction = KF.predict();
        cv::Point2f predictPt(prediction.at<float>(0), prediction.at<float>(1));

        // Si se detecta un área válida (medición confiable), corregir el filtro
        // m.m00 = área detectada; ajustar límites según el tamaño esperado del spot
        bool valid_measurement = (m.m00 > 8) && (m.m00 < 100);
        cv::Point2f measuredPt;
        if (valid_measurement) {
            int cx = static_cast<int>(m.m10 / m.m00);
            int cy = static_cast<int>(m.m01 / m.m00);
            measuredPt = cv::Point2f(cx, cy);
            measurement.at<float>(0) = cx;
            measurement.at<float>(1) = cy;
            if (!kf_initialized) {
                // Inicializar el estado del filtro con la primera medición válida
                KF.statePost.at<float>(0) = cx;
                KF.statePost.at<float>(1) = cy;
                // Inicializar velocidad con la diferencia entre la primera y segunda medición
                KF.statePost.at<float>(2) = 0;
                KF.statePost.at<float>(3) = 0;
                kf_initialized = true;
            } else {
                // No reinicializar velocidad después de la primera vez
            }
            // Corrección: el filtro ajusta su predicción usando la medición
            KF.correct(measurement);
        }

        // Visualización de los resultados del filtro de Kalman:
        // Rojo: medición directa (centro de masa)
        // Amarillo: predicción del filtro (sin corrección)
        // Verde: estimación corregida (más robusta al ruido)
        if (valid_measurement) {
            cv::circle(frame, measuredPt, 20, cv::Scalar(0, 0, 255), 2); // Medición (rojo)
        }
        cv::circle(frame, predictPt, 20, cv::Scalar(0, 255, 255), 2); // Predicción (amarillo)
        // Estimación corregida (verde)
        cv::Mat statePost = KF.statePost;
        cv::Point2f correctedPt(statePost.at<float>(0), statePost.at<float>(1));
        cv::circle(frame, correctedPt, 20, cv::Scalar(0, 255, 0), 2);

        // --- Leyenda explicativa de colores ---
        int legend_x = 20, legend_y = 30, spacing = 30;
        int radius = 8;
        int font = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.6;
        int thickness = 1;
        // Rojo: Medición directa
        cv::circle(frame, cv::Point(legend_x, legend_y), radius, cv::Scalar(0, 0, 255), -1);
        cv::putText(frame, "Medicion directa", cv::Point(legend_x + 20, legend_y + 5), font, fontScale, cv::Scalar(0,0,255), thickness, cv::LINE_AA);
        // Amarillo: Predicción
        cv::circle(frame, cv::Point(legend_x, legend_y + spacing), radius, cv::Scalar(0, 255, 255), -1);
        cv::putText(frame, "Prediccion KF", cv::Point(legend_x + 20, legend_y + spacing + 5), font, fontScale, cv::Scalar(0,255,255), thickness, cv::LINE_AA);
        // Verde: Estimación corregida
        cv::circle(frame, cv::Point(legend_x, legend_y + 2*spacing), radius, cv::Scalar(0, 255, 0), -1);
        cv::putText(frame, "Estimacion corregida", cv::Point(legend_x + 20, legend_y + 2*spacing + 5), font, fontScale, cv::Scalar(0,255,0), thickness, cv::LINE_AA);


        cv::imshow("Bright Detection", result);

        //cv::imshow("Mascara", result);

        //cv::imshow("Fondo", bgU8);
        
        // Show saturation channel?
        //cv::imshow("Saturation Channel", hsvChannels[2]);

        cv::imshow("Diferencia", diff);
        cv::imshow("Main Frame", frame);

        old_scene_mean = scene_mean;
        old_scene_stddev = scene_stddev;

        int key = cv::waitKey(30);
        if (key == 27) { // ESC
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
