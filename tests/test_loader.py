from detector.data.loaders import build_loaders

if __name__ == "__main__":
    DATA_ROOT = "data/raw/train" 
    BATCH_SIZE = 32

    print("DataLoader 파이프라인 자가 테스트를 시작합니다.")

    try:
        train_loader, val_loader = build_loaders(
            root=DATA_ROOT, 
            batch_size=BATCH_SIZE, 
            val_ratio=0.2, 
            seed=42
        )

        print(f"Train 데이터 배치 총 개수: {len(train_loader)} 개")
        print(f"Val 데이터 배치 총 개수: {len(val_loader)} 개")

        images, labels = next(iter(train_loader))

        print("\n🔍 [배치 텐서 스펙 검증]")
        print(f"   - 이미지 텐서 구조 (Shape): {images.shape}")
        print(f"   - 라벨 텐서 구조 (Shape): {labels.shape}")
        print(f"   - 이미지 데이터 타입 (Dtype): {images.dtype}")
        print(f"   - 라벨 데이터 타입 (Dtype): {labels.dtype}")

        expected_img_shape = (BATCH_SIZE, 3, 32, 32)
        expected_label_shape = (BATCH_SIZE,)

        if images.shape == expected_img_shape and labels.shape == expected_label_shape:
            print("\n🎯 [결과] 성공: '한 장씩' 주던 데이터가 모델이 먹을 수 있는 (N, C, H, W) 배칭 구조로 완벽히 묶였습니다.")
        else:
            print("\n❌ [결과] 실패: 기대한 텐서 shape와 일치하지 않습니다. 코드를 확인하세요.")

    except FileNotFoundError:
        print(f"Error: {DATA_ROOT} 경로에 데이터 셋이 준비되지 않음")
    except Exception as e:
        print(f"Edge case: {e}")
