from moviepy.editor import VideoFileClip
import video_process

output = 'project_output.mp4'

clip1 = VideoFileClip('project_video.mp4')

output_clip = clip1.fl_image(video_process.process_image) #NOTE: this function expects color images!!
output_clip.write_videofile(output, audio=False)
