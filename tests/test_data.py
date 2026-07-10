from detector.data import CifakeDataset
from torchvision import transforms

if __name__ == "__main__":
    test_transform = transforms.ToTensor()

    try:
        dataset = CifakeDataset(root="data/raw/train", transform=test_transform)
        
        print(f"✅ 데이터셋 로드 성공! 총 이미지 개수: {len(dataset)}장")
        
        if len(dataset) > 0:
            img, label = dataset[0]
            print(f"✅ 첫 번째 샘플 이미지 텐서 크기: {img.shape}") 
            print(f"✅ 첫 번째 샘플 라벨 값: {label.item()} (0: REAL, 1: FAKE)")
        else:
            print("❌ 폴더에 매칭되는 이미지가 없습니다. 경로를 확인하세요.")
            
    except FileNotFoundError as e:
        print(f"❌ 경로 오류: 'data/raw/train' 폴더가 존재하지 않거나 하위에 REAL/FAKE 폴더가 없습니다. ({e})")
