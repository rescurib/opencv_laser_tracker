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

    // EWMA learing rate
    double alpha = 0.01;
    int alpha_counter = 0;

    double threshold;
    cv::Scalar scene_mean, scene_stddev;
    cv::Scalar old_scene_mean, old_scene_stddev;
    double min_threshold = 30.0; // Umbral mínimo para evitar ruido excesivo
    cv::Scalar mean, stddev;
    cv::Mat currentFrame, grayFrame, diff;
    cv::Mat background; // CV_32F acumulador para el fondo
    cv::Mat hsvFrame;
    cv::Mat brightMask;
    cv::Mat valueMask;
    cv::Mat detectionMask;

    std::cout << "Presiona ESC para salir.\n";
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "No se pudo capturar el frame." << std::endl;
            break;
        }

        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // Calcular iluminacion media del frame.
        cv::meanStdDev(grayFrame, scene_mean, scene_stddev);

        // Inicializar el fondo con el primer frame
        if (background.empty()) 
        {
            grayFrame.convertTo(background, CV_32F);
        } else 
          {
            // Actualizar el fondo con EWMA
            double mean_diff = cv::abs(scene_mean[0] - old_scene_mean[0]);
            if(mean_diff > 50) 
            { // Si hay un cambio brusco en la iluminación, actualizar el fondo más rápido
              // durante algunos frames.
                alpha_counter = 30;
                alpha = 0.5; // Aumentar el learning rate para adaptarse rápidamente
            }

            if(alpha_counter > 0) 
            {
                alpha_counter--;
                if(alpha_counter == 0) 
                {
                    alpha = 0.01; // Volver al learning rate normal
                }
            }

            cv::accumulateWeighted(grayFrame, background, alpha);
            
          }

        // Calcular la diferencia absoluta entre el frame actual y el fondo
        cv::Mat bgU8;
        background.convertTo(bgU8, CV_8U);

        cv::GaussianBlur(grayFrame, grayFrame, cv::Size(7, 7), 1.5);
        cv::GaussianBlur(bgU8, bgU8, cv::Size(7, 7), 1.5);
        cv::absdiff(grayFrame, bgU8, diff);


        // Umbral en la media + 3*sigma de la diferencia para mantener desviaciones
        // significativas (transientes brillantes)
        cv::meanStdDev(diff, mean, stddev);
        threshold = mean[0] + 1 * stddev[0];

        // Ajustar umbral mínimo según la iluminación de la escena
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
        
        // Value masking
        cv::threshold(hsvChannels[2], valueMask, 180, 255, cv::THRESH_BINARY); // Value threshold

        //cv::bitwise_and(detectionMask, brightMask, brightMask);
        result = cv::Mat::zeros(frame.size(), frame.type());
        frame.copyTo(result, valueMask);

        // Compute moments to get center of mass of the detected bright areas
        cv::Moments m = cv::moments(valueMask, true);

        // Draw a circle at the center of mass if the area lays in a range
        if ( (m.m00 > 8) && (m.m00 < 250) ) { // Area threshold to avoid noise
            int cx = static_cast<int>(m.m10 / m.m00);
            int cy = static_cast<int>(m.m01 / m.m00);
            cv::circle(frame, cv::Point(cx, cy), 20, cv::Scalar(0, 0, 255), 2);
        }


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
