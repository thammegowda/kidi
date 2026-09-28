package ai.gowda.kidi

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

private val LightColors = lightColorScheme(
    primary = Color(0xFF126B60), onPrimary = Color.White,
    primaryContainer = Color(0xFFDDEFEA), onPrimaryContainer = Color(0xFF174F46),
    secondary = Color(0xFF77565F), onSecondary = Color.White,
    secondaryContainer = Color(0xFFF5E7EB), onSecondaryContainer = Color(0xFF63414B),
    tertiary = Color(0xFF83600C), onTertiary = Color.White,
    tertiaryContainer = Color(0xFFFFEDC5), onTertiaryContainer = Color(0xFF614606),
    background = Color(0xFFFAFBFC), onBackground = Color(0xFF202427),
    surface = Color(0xFFFFFFFF), onSurface = Color(0xFF202427),
    surfaceVariant = Color(0xFFEDF0F2), onSurfaceVariant = Color(0xFF596269),
    surfaceContainerLowest = Color.White, surfaceContainerLow = Color(0xFFF4F6F7),
    surfaceContainer = Color(0xFFEDF0F2), surfaceContainerHigh = Color(0xFFE7EBEE),
    surfaceContainerHighest = Color(0xFFE1E6E9),
    outline = Color(0xFF7C878E), outlineVariant = Color(0xFFDDE3E6),
    error = Color(0xFFB33039), onError = Color.White,
    errorContainer = Color(0xFFFCE8EA), onErrorContainer = Color(0xFF82222A),
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFF86CDBF), onPrimary = Color(0xFF00382F),
    primaryContainer = Color(0xFF223D38), onPrimaryContainer = Color(0xFFB8E5D9),
    secondary = Color(0xFFDFB8C2), onSecondary = Color(0xFF402630),
    secondaryContainer = Color(0xFF3F3037), onSecondaryContainer = Color(0xFFF0D7DE),
    tertiary = Color(0xFFE1C474), onTertiary = Color(0xFF3C2D00),
    tertiaryContainer = Color(0xFF403823), onTertiaryContainer = Color(0xFFF1DB9E),
    background = Color(0xFF131617), onBackground = Color(0xFFE6EAEC),
    surface = Color(0xFF191C1E), onSurface = Color(0xFFE6EAEC),
    surfaceVariant = Color(0xFF272D30), onSurfaceVariant = Color(0xFFABB5BC),
    surfaceContainerLowest = Color(0xFF111415), surfaceContainerLow = Color(0xFF1C2022),
    surfaceContainer = Color(0xFF24292C), surfaceContainerHigh = Color(0xFF2C3236),
    surfaceContainerHighest = Color(0xFF363D41),
    outline = Color(0xFF7E8A92), outlineVariant = Color(0xFF343C41),
    error = Color(0xFFFFABB2), onError = Color(0xFF650F1C),
    errorContainer = Color(0xFF49292F), onErrorContainer = Color(0xFFFFD8DD),
)

private val KidiFont = FontFamily(
    Font(R.font.lato_regular, FontWeight.Normal),
    Font(R.font.lato_bold, FontWeight.SemiBold),
    Font(R.font.lato_bold, FontWeight.Bold),
)

private fun type(size: Int, height: Int, weight: FontWeight = FontWeight.Normal) = TextStyle(
    fontFamily = KidiFont,
    fontWeight = weight,
    fontSize = size.sp,
    lineHeight = height.sp,
    letterSpacing = 0.sp,
)

private val KidiTypography = Typography(
    headlineLarge = type(30, 38, FontWeight.Bold),
    headlineMedium = type(26, 34, FontWeight.Bold),
    headlineSmall = type(23, 30, FontWeight.Bold),
    titleLarge = type(21, 28, FontWeight.Bold),
    titleMedium = type(16, 24, FontWeight.SemiBold),
    titleSmall = type(14, 20, FontWeight.SemiBold),
    bodyLarge = type(16, 25),
    bodyMedium = type(14, 21),
    bodySmall = type(12, 18),
    labelLarge = type(14, 20, FontWeight.SemiBold),
    labelMedium = type(12, 18, FontWeight.SemiBold),
    labelSmall = type(11, 16, FontWeight.SemiBold),
)

@Composable
internal fun KidiTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) DarkColors else LightColors,
        typography = KidiTypography,
        content = content,
    )
}