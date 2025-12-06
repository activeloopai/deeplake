from material.plugins.social.plugin import SocialPlugin as BasePlugin
import emoji
from io import BytesIO
from PIL import Image

try:
    from cairosvg import svg2png
except ImportError as e:
    import_errors.add(repr(e))
except OSError as e:
    cairosvg_error = str(e)


# Original source:
# https://github.com/squidfunk/mkdocs-material/blob/master/src/plugins/social/plugin.py
class SocialPlugin(BasePlugin):

    ## Overriding to control the description size
    def _render_card(self, site_name, title, description):
        # Render background and logo
        image = self._render_card_background((1200, 630), self.color["fill"])
        logo = self._resized_logo_promise.result()
        logo = logo.resize((logo.width * 2, logo.height * 2), Image.Resampling.LANCZOS)
        image.alpha_composite(
            logo,
            (1200 - 324, 64)
        )

        # Render page title
        font = self._get_font("Bold", 92)
        image.alpha_composite(
            self._render_text((800, 328), font, title, 3, 30),
            (64, 64)
        )

        # Render page description
        font = self._get_font("Regular", 42)
        image.alpha_composite(
            self._render_text((1200-228, 120), font, description, 3, 21),
            (64, 488)
        )

        # Return social card image
        return image

    def _render_text(self, size, font, text, lmax, spacing = 0):
        return super()._render_text(size, font, emoji.replace_emoji(text, "").strip(), lmax, spacing)
