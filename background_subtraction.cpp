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
    const double alpha = 0.025;

    cv::Mat currentFrame, grayFrame, diff;
    cv::Mat background; // CV_32F acumulador para el fondo

    std::cout << "Presiona ESC para salir.\n";
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "No se pudo capturar el frame." << std::endl;
            break;
        }

        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // Inicializar el fondo con el primer frame
        if (background.empty()) {
            grayFrame.convertTo(background, CV_32F);
        } else {
            // Actualizar el fondo con EWMA
            cv::accumulateWeighted(grayFrame, background, alpha);
        }

        // Calcular la diferencia absoluta entre el frame actual y el fondo
        cv::Mat bgU8;
        background.convertTo(bgU8, CV_8U);
        cv::absdiff(grayFrame, bgU8, diff);

        // Umbral en la media + 3*sigma de la diferencia para mantener desviaciones
        // significativas (transientes brillantes)
        cv::Scalar mean, stddev;
        cv::meanStdDev(diff, mean, stddev);
        double threshold = mean[0] + 4 * stddev[0];
        cv::Mat foregroundMask;
        cv::threshold(diff, foregroundMask, threshold, 255, cv::THRESH_BINARY);

        // Mostrar resultado enmascarado en el frame con color original
        cv::Mat result = cv::Mat::zeros(frame.size(), frame.type());
        frame.copyTo(result, foregroundMask);

        cv::imshow("Mascara", result);

        cv::imshow("Fondo", bgU8);

        cv::imshow("Diferencia", diff);


        int key = cv::waitKey(30);
        if (key == 27) { // ESC
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
