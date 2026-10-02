# Results

PSNR/SSIM compare each rendering with the photo, so they reward copying pixels, not simplifying. Read them next to the path count and file size.

## canal

| method | paths | KB | PSNR | SSIM |
|---|---|---|---|---|
| vtracer (defaults) | 14985 | 16433 | 23.0 | 0.767 |
| vtracer (matched paths) | 112 | 1489 | 18.6 | 0.482 |
| SuperSVG (CVPR 2024) | 109 | 60 | 18.0 | 0.409 |
| segvec (automatic) | 122 | 56 | 15.4 | 0.365 |
| segvec (--care) | 126 | 56 | 15.6 | 0.363 |

## delft-street

| method | paths | KB | PSNR | SSIM |
|---|---|---|---|---|
| vtracer (defaults) | 12652 | 12215 | 24.1 | 0.765 |
| vtracer (matched paths) | 213 | 2287 | 18.9 | 0.503 |
| SuperSVG (CVPR 2024) | 213 | 116 | 18.1 | 0.396 |
| segvec (automatic) | 155 | 71 | 15.3 | 0.375 |
| segvec (--care) | 234 | 79 | 17.1 | 0.383 |

## hikers

| method | paths | KB | PSNR | SSIM |
|---|---|---|---|---|
| vtracer (defaults) | 17699 | 19404 | 24.5 | 0.781 |
| vtracer (matched paths) | 112 | 2097 | 19.1 | 0.492 |
| SuperSVG (CVPR 2024) | 113 | 62 | 18.3 | 0.377 |
| segvec (automatic) | 139 | 89 | 15.5 | 0.356 |
| segvec (--care) | 126 | 74 | 15.5 | 0.356 |

## lone-hiker

| method | paths | KB | PSNR | SSIM |
|---|---|---|---|---|
| vtracer (defaults) | 11742 | 13745 | 19.6 | 0.73 |
| vtracer (matched paths) | 77 | 1172 | 17.2 | 0.516 |
| SuperSVG (CVPR 2024) | 75 | 41 | 19.4 | 0.462 |
| segvec (automatic) | 85 | 55 | 16.3 | 0.444 |
| segvec (--care) | 82 | 51 | 17.1 | 0.442 |

## mountain-lake

| method | paths | KB | PSNR | SSIM |
|---|---|---|---|---|
| vtracer (defaults) | 9127 | 10415 | 22.6 | 0.751 |
| vtracer (matched paths) | 99 | 758 | 13.3 | 0.563 |
| SuperSVG (CVPR 2024) | 97 | 53 | 21.7 | 0.577 |
| segvec (automatic) | 63 | 41 | 18.3 | 0.566 |
| segvec (--care) | 105 | 55 | 19.4 | 0.567 |

## oostpoort

| method | paths | KB | PSNR | SSIM |
|---|---|---|---|---|
| vtracer (defaults) | 18237 | 20325 | 23.3 | 0.75 |
| vtracer (matched paths) | 81 | 1607 | 16.1 | 0.415 |
| SuperSVG (CVPR 2024) | 81 | 44 | 17.6 | 0.379 |
| segvec (automatic) | 70 | 55 | 16.0 | 0.366 |
| segvec (--care) | 94 | 64 | 16.6 | 0.37 |

## Average over 6 photos

| method | paths | KB | PSNR | SSIM |
|---|---|---|---|---|
| vtracer (defaults) | 14074 | 15423 | 22.8 | 0.76 |
| vtracer (matched paths) | 116 | 1568 | 17.2 | 0.50 |
| SuperSVG (CVPR 2024) | 115 | 63 | 18.8 | 0.43 |
| segvec (automatic) | 106 | 61 | 16.1 | 0.41 |
| segvec (--care) | 128 | 63 | 16.9 | 0.41 |
