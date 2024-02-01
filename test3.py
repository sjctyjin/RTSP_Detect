import vlc

# 创建VLC播放器实例
vlc_instance = vlc.Instance('--no-xlib')

# 创建媒体对象，指定RTSP流的URL
media = vlc_instance.media_new('rtsp://192.168.1.105/stream1')

# 创建媒体播放器
player = vlc_instance.media_player_new()

# 设置媒体
player.set_media(media)

# 开始播放
player.play()

# 一直播放，直到手动停止
input("Press Enter to stop...")
player.stop()
