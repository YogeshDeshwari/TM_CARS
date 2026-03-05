from style_compiler import StyleCompiler, make_recipe


def main():
    compiler = StyleCompiler(size=2048)

    recipes = [
        make_recipe(
            name="Nebula_Novel_01",
            seed=1337,
            base=(10, 10, 18),
            secondary=(30, 10, 70),
            accent=(80, 0, 200),
            highlight=(220, 240, 255),
            motif="galaxy",
            gloss_base="satin",
            accent_gloss_boost=210,
            dirt_amount=0.25,
        ),
        make_recipe(
            name="Autumn_Novel_01",
            seed=20261,
            base=(30, 20, 12),
            secondary=(110, 55, 25),
            accent=(180, 95, 40),
            highlight=(235, 220, 200),
            motif="leaves",
            gloss_base="matte",
            accent_gloss_boost=160,
            dirt_amount=0.45,
        ),
        make_recipe(
            name="Minimal_Esports_01",
            seed=99,
            base=(14, 14, 16),
            secondary=(32, 32, 36),
            accent=(235, 235, 235),
            highlight=(255, 60, 60),
            motif="minimal_blocks",
            gloss_base="satin",
            accent_gloss_boost=200,
            dirt_amount=0.20,
        ),
    ]

    for r in recipes:
        engine = compiler.render(r, team_name=r.name)
        engine.set_dirt_amount(r.dirt_amount)
        engine.save()


if __name__ == "__main__":
    main()

