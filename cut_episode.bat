$videoInput = "screenshots/input_omydays.mp4"
$videoOutput = "screenshots/input_omydays_cutted.mp4"
$aPercent = 30
$bPercent = 80

$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$durationText = ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $videoInput
$duration = [double]::Parse($durationText.Trim(), $invariant)

$startSeconds = $duration * $aPercent / 100
$clipSeconds = $duration * ($bPercent - $aPercent) / 100

$startArg = $startSeconds.ToString("0.######", $invariant)
$lengthArg = $clipSeconds.ToString("0.######", $invariant)

ffmpeg -y -ss $startArg -i $videoInput -t $lengthArg `
  -c:v libx264 -preset fast -crf 18 `
  -c:a aac -b:a 192k `
  $videoOutput