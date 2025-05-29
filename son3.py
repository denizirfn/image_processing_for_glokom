import cv2
import numpy as np


def correct_illumination(image):
    """
    Morfolojik açılma kullanarak görüntü aydınlatmasını düzeltir.

    Args:
        image (numpy.ndarray): Giriş görüntüsü.

    Returns:
        numpy.ndarray: Aydınlatma düzeltilmiş görüntü.
    """
    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Morfolojik açılma ile arka plan tahminini yap
    kernel_size = min(gray.shape) // 10  # Görüntü boyutunun %10'u
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Arka planı çıkar ve kontrastı artır
    foreground = cv2.subtract(gray, background)

    # DC seviyesi ekle
    corrected = cv2.add(foreground, 50)

    # Gri tonlamadan renge çevir
    corrected_color = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)

    return corrected_color


def remove_blood_vessels(image):
    """
    Kan damarlarını çıkarır ve eksik bölgeleri komşu piksel değerleri ile doldurur.

    Args:
        image (numpy.ndarray): Giriş görüntüsü.

    Returns:
        numpy.ndarray: Kan damarları çıkarılmış görüntü.
    """
    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE ile kontrastı artır
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced_gray = clahe.apply(gray)

    # Kan damarlarını tespit etmek için morfolojik işlemler
    kernel = np.ones((3, 3), np.uint8)
    vessel_mask = cv2.morphologyEx(enhanced_gray, cv2.MORPH_GRADIENT, kernel)

    # Eşikleme ile damar maskesini oluştur
    _, vessel_mask = cv2.threshold(vessel_mask, 19, 255, cv2.THRESH_BINARY)

    # Kan damarlarını çıkar ve eksik pikselleri doldur
    vessel_removed_image = cv2.inpaint(image, vessel_mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)

    return vessel_removed_image, vessel_mask


def inpaint_vessels(image, vessel_mask):
    """
    Inpainting işlemi

    Args:
        image (numpy.ndarray): Görüntü.
        vessel_mask (numpy.ndarray): Kan damarlarını gösteren maske.

    Returns:
        numpy.ndarray: Inpaint edilmiş görüntü.
    """
    inpainted_image = cv2.inpaint(image, vessel_mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)
    return inpainted_image


def multi_thresholding(image):
    """
    Multi-thresholding tekniği ile optik disk ve cup segmentasyonu yapar.

    Args:
        image (numpy.ndarray): Giriş görüntüsü.

    Returns:
        tuple: (disc_mask, cup_mask) - Optik disk ve cup maskeleri.
    """
    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Görüntüyü normalize et
    gray_normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Yüksek eşik değeri ile optik cup'ı tespit et
    _, cup_mask = cv2.threshold(gray_normalized, 150, 255, cv2.THRESH_BINARY)

    # Düşük eşik değeri ile optik disk'i tespit et
    _, disc_mask = cv2.threshold(gray_normalized, 100, 255, cv2.THRESH_BINARY)

    return disc_mask, cup_mask


def calculate_disc_cup_ratio(disc_mask, cup_mask):
    """
    Optik disk ve optik cup arasındaki alan oranını hesaplar.

    Args:
        disc_mask (numpy.ndarray): Optik disk maskesi.
        cup_mask (numpy.ndarray): Optik cup maskesi.

    Returns:
        float: Disk ve cup arasındaki oran.
    """
    # Disk ve cup maskelerindeki beyaz alanları say
    disc_area = np.sum(disc_mask == 255)
    cup_area = np.sum(cup_mask == 255)

    # Eğer disk alanı sıfırsa oran hesaplamadan önce kontrol et
    if disc_area == 0:
        return 0

    # Disc/Cup oranını hesapla
    ratio = cup_area / disc_area
    return ratio


def glaucoma_detection(disc_mask, cup_mask):
    """
    Glokom şüphesi olup olmadığını disk/cup oranına göre kontrol eder.

    Args:
        disc_mask (numpy.ndarray): Optik disk maskesi.
        cup_mask (numpy.ndarray): Optik cup maskesi.

    Returns:
        str: Glokom durumu (şüpheli veya değil).
    """
    ratio = calculate_disc_cup_ratio(disc_mask, cup_mask)
    print(f"Disk-Cup Oranı: {ratio}")

    # Glokom şüphesi için eşiği belirle (örneğin, 0.5)
    if ratio > 0.5:
        return "Glokom şüphesi yüksek."
    else:
        return "Glokom şüphesi yok."


def main():
    # Görüntüyü oku
    image = cv2.imread('glokom.jpg')

    if image is None:
        print("Görüntü yüklenemedi. Dosya yolunu kontrol edin.")
        return

    # Aydınlatma düzeltmesini uygula
    corrected_image = correct_illumination(image)

    # Kan damarlarını çıkar
    vessel_removed_image, vessel_mask = remove_blood_vessels(corrected_image)

    # Kan damarlarını inpaint et
    inpainted_image = inpaint_vessels(corrected_image, vessel_mask)

    # Multi-thresholding ile optik disk ve cup segmentasyonu yap
    disc_mask, cup_mask = multi_thresholding(inpainted_image)

    # Glokom tespiti yap
    glaucoma_status = glaucoma_detection(disc_mask, cup_mask)
    print(glaucoma_status)

    # Sonuçları kaydet
    cv2.imshow('Original Görüntü', image)
    cv2.imshow('Düzgün Aydınlatılmış Görüntü', corrected_image)
    cv2.imshow('Kan Damarları Çıkarılmış Görüntü', vessel_removed_image)
    cv2.imshow('Inpainted Görüntü', inpainted_image)
    cv2.imshow('Optik Disk Maskesi', disc_mask)
    cv2.imshow('Optik Cup Maskesi', cup_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
