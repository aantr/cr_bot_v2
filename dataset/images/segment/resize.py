import os
from PIL import Image
import glob

def resize_images_recursive(scale_factor, create_backup=False):
    """
    Рекурсивно изменяет размер всех изображений в текущей директории и всех поддиректориях
    
    Args:
        scale_factor (float): Коэффициент изменения размера
        create_backup (bool): Создавать ли резервные копии
    """
    
    # Получаем путь к корневой директории скрипта
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Поддерживаемые форматы изображений
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    
    # Находим все изображения рекурсивно
    image_files = []
    for extension in image_extensions:
        pattern = os.path.join(root_dir, '**', extension)
        image_files.extend(glob.glob(pattern, recursive=True))
        
        pattern_upper = os.path.join(root_dir, '**', extension.upper())
        image_files.extend(glob.glob(pattern_upper, recursive=True))
    
    # Убираем дубликаты (на случай если файлы с разным регистром совпадают)
    image_files = list(set(image_files))
    
    if not image_files:
        print("В директории и поддиректориях не найдено изображений.")
        return
    
    print(f"Найдено {len(image_files)} изображений в директории и поддиректориях:")
    
    # Группируем по директориям для красивого вывода
    dir_files = {}
    for file_path in image_files:
        dir_name = os.path.dirname(file_path)
        if dir_name not in dir_files:
            dir_files[dir_name] = []
        dir_files[dir_name].append(file_path)
    
    for dir_name, files in dir_files.items():
        print(f"\n{dir_name}:")
        for file_path in files:
            print(f"  - {os.path.basename(file_path)}")
    
    print(f"\nИзменение размера всех изображений в {scale_factor} раз...")
    
    success_count = 0
    error_count = 0
    
    for image_path in image_files:
        try:
            # Открываем изображение
            with Image.open(image_path) as img:
                # Получаем текущие размеры
                width, height = img.size
                
                # Вычисляем новые размеры
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                if create_backup:
                    # Создаем резервную копию
                    name, ext = os.path.splitext(image_path)
                    backup_path = f"{name}_original{ext}"
                    img.save(backup_path)
                    print(f"  Создана резервная копия: {os.path.basename(backup_path)}")
                
                # Изменяем размер
                resized_img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
                
                # Сохраняем изображение
                resized_img.save(image_path)
                
                # Выводим относительный путь от корневой директории
                rel_path = os.path.relpath(image_path, root_dir)
                print(f"✓ {rel_path}: {width}x{height} → {new_width}x{new_height}")
                success_count += 1
                
        except Exception as e:
            rel_path = os.path.relpath(image_path, root_dir)
            print(f"✗ Ошибка при обработке {rel_path}: {str(e)}")
            error_count += 1
    
    print(f"\nГотово! Успешно обработано: {success_count}, ошибок: {error_count}")

def main():
    try:
        print("Рекурсивное изменение размера изображений")
        print("=========================================")
        
        # Запрашиваем коэффициент изменения у пользователя
        k = float(input("Введите коэффициент изменения размера (например, 0.5 для уменьшения в 2 раза, 2 для увеличения в 2 раза): "))
        
        if k <= 0:
            print("Коэффициент должен быть положительным числом.")
            return
        
        # Спрашиваем про резервные копии
        backup_choice = input("Создавать резервные копии оригиналов? (y/n): ").lower().strip()
        create_backup = backup_choice in ['y', 'yes', 'д', 'да']
        
        if create_backup:
            print("Резервные копии будут созданы с суффиксом '_original'")
        
        print("\nНачинаем обработку...")
        resize_images_recursive(k, create_backup)
        
    except ValueError:
        print("Пожалуйста, введите корректное число.")
    except KeyboardInterrupt:
        print("\nОперация прервана пользователем.")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {str(e)}")

if __name__ == "__main__":
    main()