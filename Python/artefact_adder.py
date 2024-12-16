import cv2
import numpy as np

def gaussian_noise(frame, noise_level):
    noise = np.zeros_like(frame, dtype=np.uint8)
    cv2.randn(noise, (0, 0, 0), (noise_level, noise_level, noise_level))
    noisy_frame = cv2.add(frame, noise)

    return noisy_frame


def compression_artifacts(frame, compression_quality):
    # Encode l’image à l’aide du format spécifié (JPEG) selon la qualité de compression passée
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality]
    result, encimg = cv2.imencode('.jpg', frame, encode_param)
    compressed_frame = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    # Retourne l'image compressée
    return compressed_frame


def motion_blur(frame, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size

    blurred_frame = cv2.filter2D(frame, -1, kernel)
    return blurred_frame


def color_banding(frame, levels):
    quantized_frame = (frame // (256 // levels)) * (256 // levels)
    return quantized_frame


def salt_and_pepper_noise(frame, amount):
    noisy_frame = frame.copy()
    num_salt = np.ceil(amount * frame.size * 0.5)
    num_pepper = np.ceil(amount * frame.size * 0.5)
    
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in frame.shape]
    noisy_frame[coords[0], coords[1], :] = 255

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in frame.shape]
    noisy_frame[coords[0], coords[1], :] = 0
    
    return noisy_frame


def lens_distortion(frame, k1, k2):
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w/2],
                  [0, w, h/2],
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([k1, k2, 0, 0], dtype=np.float32)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
    distorted_frame = cv2.undistort(frame, K, D, None, new_K)
    return distorted_frame


def vignetting(frame):
    h, w = frame.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(w, w/2)
    Y_resultant_kernel = cv2.getGaussianKernel(h, h/2)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()
    vignette_frame = np.empty_like(frame)
    for i in range(3):
        vignette_frame[:, :, i] = frame[:, :, i] * mask
    return vignette_frame


def chromatic_aberration(frame, shift):
    b, g, r = cv2.split(frame)
    b = np.roll(b, shift, axis=1)
    r = np.roll(r, -shift, axis=1)
    aberrated_frame = cv2.merge([b, g, r])
    return aberrated_frame


def dust_dirt_particles(frame, num_particles):
    dusty_frame = frame.copy()
    h, w, _ = frame.shape
    for _ in range(num_particles):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        radius = np.random.randint(5, 15)
        color = (np.random.randint(50, 100),) * 3
        cv2.circle(dusty_frame, (x, y), radius, color, -1)
    return dusty_frame


def simulate_frame_dropping(frame, out, drop_rate):
    if np.random.rand() > drop_rate:
        out.write(frame)


def add_artifacts(input_video_path,             # Path de la vidéo
                  noise_level=80,               # La quantité de bruit gaussien
                  compression_quality=1,        # La qualité de la compression JPEG
                  kernel_size=15,               # La grosseur du kernel
                  color_branding_levels=8,      # Niveaux de marquage des couleurs
                  salt_and_pepper_amount=0.01,  # La quantité de bruit sel et poivre
                  lens_k1=-0.5,                 # Coefficient 1 de la distortion des lentilles
                  lens_k2=0.0,                  # Coefficient 2 de la distortion des lentilles
                  chromatic_aberration_shift=2, # Décalage d'aberration chromatique
                  dust_particles=10,            # Le nombre de particules de poussière à créer
                  drop_rate=0.1):               # Le taux de chute des frames
    
    # Va aller ouvrir le fichier et créé un objet associé à la vidéo
    cap = cv2.VideoCapture(input_video_path)

    # Si aucun fichier est ouvert, quitte le programme 
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Largeur
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Hauteur
    fps    = cap.get(cv2.CAP_PROP_FPS)                # FPS

    # Type de fichier
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # Bruit gaussien
    out_gaussian_noise = cv2.VideoWriter('gaussian_noise.mp4', fourcc, fps, (width, height))
    # Compression
    out_compression = cv2.VideoWriter('compression.mp4', fourcc, fps, (width, height))
    # Motion blur
    out_motion_blur = cv2.VideoWriter('motion_blur.mp4', fourcc, fps, (width, height))
    # Color branding
    out_color_banding = cv2.VideoWriter('color_banding.mp4', fourcc, fps, (width, height))
    # Bruit sel et poivre
    out_salt_pepper_noise = cv2.VideoWriter('salt_pepper_noise.mp4', fourcc, fps, (width, height))
    # Distortion de lentilles
    out_lens_distortion = cv2.VideoWriter('lens_distortion.mp4', fourcc, fps, (width, height))
    # Vignette
    out_vignetting = cv2.VideoWriter('vignetting.mp4', fourcc, fps, (width, height))
    # Abérattion chromatique
    out_chromatic_aberration = cv2.VideoWriter('chromatic_aberration.mp4', fourcc, fps, (width, height))
    # Particules de poussières
    out_dust_particles = cv2.VideoWriter('dust_particles.mp4', fourcc, fps, (width, height))
    # Frame drops
    out_frame_dropping = cv2.VideoWriter('frame_dropping.mp4', fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames to add artifacts...")

    frame_number = 0
    saved_images = False  # Flag to save the first frame with artifacts

    # Tant que la vidéo n'est pas finie
    while True:
        # Lire l'image suivante
        ret, frame = cap.read()

        # Si l'image n'existe pas, sort du while
        if not ret:
            break

        # Ajoute le bruit gaussien
        noisy_frame = gaussian_noise(frame, noise_level)
        out_gaussian_noise.write(noisy_frame)
        if not saved_images:
            cv2.imwrite("gaussian_noise.png", noisy_frame)

        # Ajoute de la compression
        compressed_frame = compression_artifacts(frame, compression_quality)
        out_compression.write(compressed_frame)
        if not saved_images:
            cv2.imwrite("compression.png", compressed_frame)

        # Ajoute du flou de mouvement
        blurred_frame = motion_blur(frame, kernel_size)
        out_motion_blur.write(blurred_frame)
        if not saved_images:
            cv2.imwrite("motion_blur.png", blurred_frame)

        # Ajoute du color branding
        banded_frame = color_banding(frame, color_branding_levels)
        out_color_banding.write(banded_frame)
        if not saved_images:
            cv2.imwrite("color_banding.png", banded_frame)

        # Ajoute du bruit sel et poivre
        sp_noise_frame = salt_and_pepper_noise(frame, salt_and_pepper_amount)
        out_salt_pepper_noise.write(sp_noise_frame)
        if not saved_images:
            cv2.imwrite("salt_pepper_noise.png", sp_noise_frame)

        # Ajoute des distortions de lentilles
        distorted_frame = lens_distortion(frame, lens_k1, lens_k2)
        out_lens_distortion.write(distorted_frame)
        if not saved_images:
            cv2.imwrite("lens_distortion.png", distorted_frame)

        # Ajoute une vignette
        vignette_frame = vignetting(frame)
        out_vignetting.write(vignette_frame)
        if not saved_images:
            cv2.imwrite("vignetting.png", vignette_frame)

        # Ajoute de l'abérration chromatique 
        aberrated_frame = chromatic_aberration(frame, chromatic_aberration_shift)
        out_chromatic_aberration.write(aberrated_frame)
        if not saved_images:
            cv2.imwrite("chromatic_aberration.png", aberrated_frame)

        # Ajoute des particules de poussières
        dusty_frame = dust_dirt_particles(frame, dust_particles)
        out_dust_particles.write(dusty_frame)
        if not saved_images:
            cv2.imwrite("dust_particles.png", dusty_frame)

        # Simule des baisses de FPS
        simulate_frame_dropping(frame, out_frame_dropping, drop_rate)

        if not saved_images:
            saved_images = True  # Sauvegarde les premières images

        frame_number += 1
        if frame_number % 10 == 0:
            print(f"Processed {frame_number}/{frame_count} frames")

    # Release nos objets
    cap.release()
    out_gaussian_noise.release()
    out_compression.release()
    out_motion_blur.release()
    out_color_banding.release()
    out_salt_pepper_noise.release()
    out_lens_distortion.release()
    out_vignetting.release()
    out_chromatic_aberration.release()
    out_dust_particles.release()
    out_frame_dropping.release()
    print(f"Artifact addition complete")

if __name__ == "__main__":
    input_video = "input_video.mp4"
    add_artifacts(input_video)
