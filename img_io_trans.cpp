
#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define  IMG_NOR_W 640
#define  IMG_NOR_H 480
#define  MORPH_SIZE 5
int fiber_inimg_normalize(Mat fiber_img_raw, Mat* fiber_img_rgb, Mat* fiber_img_gray)

{
	if (fiber_img_raw.rows != IMG_NOR_H || fiber_img_raw.cols != IMG_NOR_W)//若读入图片不是640x480的标准图片，则将输入图片resize成640x480的图片
	{
		Size fiber_norm_size;
		fiber_norm_size.width = IMG_NOR_W;
		fiber_norm_size.height = IMG_NOR_H;
		resize(fiber_img_raw, *fiber_img_rgb, fiber_norm_size);

	}
	else
		*fiber_img_rgb = fiber_img_raw;

	cvtColor(*fiber_img_rgb, *fiber_img_gray, CV_BGR2GRAY);//RGB图转灰度图

	return 0;
}

int fiber_err_proc(int err_flag)
{
	if (err_flag == 0)
		return 0;
	else if ((err_flag & 0x001) == 1)//图片模糊，不进行后续处理，退出程序
	{
		printf("grad detect module have got a indistinct image,so exit the program.\n");
		return -1;
	}
	else if ((err_flag & 0x002) == 1)//无法定位圆，退出程序
	{
		printf("center finder error.\n");
		return -1;
	}

	return 0;
}

int img_enhance(Mat* fiber_img_gray)
{
	Mat fiber_morph_element = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * MORPH_SIZE + 1, 2 * MORPH_SIZE + 1),
		Point(MORPH_SIZE, MORPH_SIZE));
	Mat fiber_img_tophat;
	morphologyEx(*fiber_img_gray, fiber_img_tophat, MORPH_TOPHAT, fiber_morph_element, Point(-1, -1), 2);
	fiber_img_tophat = fiber_img_tophat + *fiber_img_gray;
	Mat fiber_img_bkhat;
	morphologyEx(*fiber_img_gray, fiber_img_bkhat, MORPH_BLACKHAT, fiber_morph_element, Point(-1, -1), 2);
	fiber_img_bkhat = fiber_img_tophat - fiber_img_bkhat;
	*fiber_img_gray = fiber_img_bkhat;
	fiber_img_tophat.release();
	fiber_img_bkhat.release();
	return 0;
}


int img_getsobelmap(Mat input_map, Mat* sobel_map)
{
	Mat src = input_map;
	Mat dst = *sobel_map;


	int offset = 1;
	int sobel_x = 0;
	int sobel_y = 0;
	double sobel = 0;

	int prewitt = 0;
	int prewitt_i = 0;
	double pret = 0;

	for (int i = offset; i < src.rows - offset; i++)
		for (int j = offset; j < src.cols - offset; j++)
		{

			//计算sobel
			sobel_y = src.at<uchar>(i - 1 * offset, j + 1 * offset)
				+ 2 * src.at<uchar>(i, j + 1 * offset)
				+ src.at<uchar>(i + 1 * offset, j + 1 * offset)
				- src.at<uchar>(i - 1 * offset, j - 1 * offset)
				- 2 * src.at<uchar>(i, j - 1 * offset)
				- src.at<uchar>(i + 1 * offset, j - 1 * offset);

			sobel_x = src.at<uchar>(i - 1 * offset, j - 1 * offset)
				+ 2 * src.at<uchar>(i - 1 * offset, j)
				+ src.at<uchar>(i - 1 * offset, j + 1 * offset)
				- src.at<uchar>(i + 1 * offset, j - 1 * offset)
				- 2 * src.at<uchar>(i + 1 * offset, j)
				- src.at<uchar>(i + 1 * offset, j + 1 * offset);


			//计算prewitt
			prewitt = src.at<uchar>(i, j - 1 * offset)
				+ 1 * src.at<uchar>(i + 1 * offset, j - 1 * offset)
				+ src.at<uchar>(i + 1 * offset, j)
				- src.at<uchar>(i - 1 * offset, j)
				- 1 * src.at<uchar>(i - 1 * offset, j + 1 * offset)
				- src.at<uchar>(i, j + 1 * offset);

			prewitt_i = src.at<uchar>(i + 1 * offset, j)
				+ 1 * src.at<uchar>(i + 1 * offset, j + 1 * offset)
				+ src.at<uchar>(i, j + 1 * offset)
				- src.at<uchar>(i - 1 * offset, j - 1 * offset)
				- 1 * src.at<uchar>(i, j - 1 * offset)
				- src.at<uchar>(i - 1 * offset, j);



			sobel = sqrt(double(sobel_x*sobel_x + sobel_y*sobel_y));
			pret = abs(prewitt) + abs(prewitt_i);

			dst.at<float>(i, j) = (float)(sobel + pret);
		}


	return 0;
}
