<template>
    <div class="tooltip">
        <slot></slot>
    </div>
</template>

<script>
export default {
    name: 'Tooltip',
    props: ['title']
}
</script>
